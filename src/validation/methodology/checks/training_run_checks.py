"""Static checks comparing an experiment with the training run of the loaded model.

Both checks read the config snapshot saved next to the model (see ``training_run_config``).
Without a saved snapshot nothing can be compared, so they report nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.validation.methodology.check_registry import CheckContext, Finding, methodology_check
from src.validation.methodology.checks.training_run_config import load_training_run_config
from src.validation.methodology.condition_evaluator import MISSING, resolve_path

SELECTION_KEYS: tuple[str, ...] = ("subject_ids", "session_ids", "run_ids")


@methodology_check("experiment_data_overlaps_training_data")
def experiment_data_overlaps_training_data(check_context: CheckContext) -> list[Finding]:
    """The experiment must not evaluate recordings from the source the model was trained on.

    Only provable overlaps are reported: the same dataset with intersecting subject, session and run
    selections, where ``null`` selects all available ids. A synthetic source overlaps only when its whole
    generator configuration is the same, because only then it produces the same data.
    """
    model_path = Path(check_context.config.model_path.path)
    training_config = load_training_run_config(model_path)
    if training_config is None:
        return []

    training_source: Mapping[str, Any] = training_config.get("source", {})
    experiment_source: Mapping[str, Any] = check_context.config.source.model_dump(mode="json")
    if not are_sources_overlapping(training_source, experiment_source):
        return []

    return [
        Finding(
            detail=(
                f"The model '{model_path}' was trained on {describe_source(training_source)}; "
                f"the experiment evaluates {describe_source(experiment_source)}, which selects the same recordings."
            ),
            location="source",
        )
    ]


@methodology_check("experiment_preprocessing_matches_training")
def experiment_preprocessing_matches_training(check_context: CheckContext) -> list[Finding]:
    """Settings listed in ``params.paths`` must be the same as in the training run of the loaded model.

    Disabled blocks (``enabled: false`` on both sides) are equal whatever their other parameters are.
    A path missing from the training snapshot (older config schema) cannot be compared and is skipped.

    Params:
        paths: dot-separated config paths to compare, e.g. ``paradigm.filter``.
    """
    model_path = Path(check_context.config.model_path.path)
    training_config = load_training_run_config(model_path)
    if training_config is None:
        return []

    experiment_config = check_context.config.model_dump(mode="json")
    findings: list[Finding] = []
    for path in check_context.params.get("paths", []):
        training_value = resolve_path(training_config, path)
        if training_value is MISSING:
            continue

        experiment_value = resolve_path(experiment_config, path)
        if are_settings_equivalent(training_value, experiment_value):
            continue

        findings.append(
            Finding(
                detail=(
                    f"'{path}' is {experiment_value} in the experiment, "
                    f"but the model '{model_path}' was trained with {training_value}."
                ),
                location=path,
            )
        )
    return findings


def are_sources_overlapping(training_source: Mapping[str, Any], experiment_source: Mapping[str, Any]) -> bool:
    """Return whether both source configs provably select at least one common recording."""
    backend = experiment_source.get("backend")
    if training_source.get("backend") != backend:
        return False

    if backend == "synthetic":
        return training_source == experiment_source
    if backend == "external":
        is_same_dataset = training_source.get("name") == experiment_source.get("name")
        return is_same_dataset and are_selections_overlapping(training_source, experiment_source, SELECTION_KEYS)
    if backend == "filesystem":
        is_same_directory = training_source.get("path") == experiment_source.get("path")
        return is_same_directory and are_selections_overlapping(training_source, experiment_source, ("subject_ids",))
    return False


def are_selections_overlapping(
        training_source: Mapping[str, Any],
        experiment_source: Mapping[str, Any],
        selection_keys: Sequence[str],
) -> bool:
    """Return whether the id selections overlap for every key (subjects, sessions, runs)."""
    for selection_key in selection_keys:
        training_ids = training_source.get(selection_key)
        experiment_ids = experiment_source.get(selection_key)
        if not are_id_selections_overlapping(training_ids, experiment_ids):
            return False
    return True


def are_id_selections_overlapping(first_ids: Sequence[object] | None, second_ids: Sequence[object] | None) -> bool:
    """Return whether two id selections share an id; ``None`` selects all available ids.

    Ids are compared as text, as the data loaders do (``1`` and ``"1"`` are the same session).
    """
    if first_ids is None or second_ids is None:
        return True

    first_id_texts = {str(identifier) for identifier in first_ids}
    return any(str(identifier) in first_id_texts for identifier in second_ids)


def are_settings_equivalent(training_value: object, experiment_value: object) -> bool:
    """Return whether two config values configure the same preprocessing."""
    are_both_disabled = is_disabled_block(training_value) and is_disabled_block(experiment_value)
    if are_both_disabled:
        return True
    return training_value == experiment_value


def is_disabled_block(value: object) -> bool:
    """Return whether ``value`` is a config block with ``enabled: false``."""
    return isinstance(value, Mapping) and value.get("enabled") is False


def describe_source(source: Mapping[str, Any]) -> str:
    """Return a short human-readable description of the data a source config selects."""
    dataset = source.get("name") or source.get("path") or source.get("backend")
    selections = ", ".join(f"{selection_key}={source.get(selection_key)}" for selection_key in SELECTION_KEYS)
    return f"'{dataset}' ({selections})"
