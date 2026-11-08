"""Static methodology checks evaluated from the experiment configuration only.

Each check is referenced from ``configs/methodology/rules.yaml`` by its
registered name. Checks return a list of findings; an empty list means the
rule is satisfied.
"""

from __future__ import annotations

from src.types.dto.config.protocol_config import Generalization, Usage
from src.validation.methodology.check_registry import CheckContext, Finding, methodology_check


@methodology_check("split_matches_protocol")
def split_matches_protocol(ctx: CheckContext) -> list[Finding]:
    """The split strategy must correspond to the declared generalization.

    Params:
        compatible_splits: mapping ``generalization -> list of allowed split backends``.
    """
    generalization = ctx.config.protocol.generalization.value
    split_backend = ctx.config.split.backend
    compatible: dict[str, list[str]] = ctx.params.get("compatible_splits", {})
    allowed = compatible.get(generalization, [])

    if split_backend in allowed:
        return []

    allowed_text = ", ".join(allowed) if allowed else "none of the available splitters"
    return [
        Finding(
            detail=f"Declared generalization '{generalization}' requires split backend {allowed_text}, but '{split_backend}' is configured.",
            location="split.backend",
        )
    ]


def _known_count(values: list | None) -> int | None:
    """Number of explicitly selected ids, ``None`` when 'all available' is selected."""
    return None if values is None else len(values)


@methodology_check("enough_groups_for_protocol")
def enough_groups_for_protocol(ctx: CheckContext) -> list[Finding]:
    """Cross-session needs >= 2 sessions, cross-subject needs >= 2 subjects.

    Only provable cases are reported: an explicit id list that is too short,
    or the synthetic source (always one subject and one session). ``null``
    (= all available) is not reported.

    Params:
        min_sessions: minimum number of sessions for cross-session (default 2).
        min_subjects: minimum number of subjects for cross-subject (default 2).
    """
    generalization = ctx.config.protocol.generalization
    source = ctx.config.source
    min_sessions = int(ctx.params.get("min_sessions", 2))
    min_subjects = int(ctx.params.get("min_subjects", 2))

    if source.backend == "synthetic":
        n_subjects: int | None = 1
        n_sessions: int | None = 1
    else:
        n_subjects = _known_count(getattr(source, "subject_ids", None))
        n_sessions = _known_count(getattr(source, "session_ids", None))

    findings: list[Finding] = []
    if generalization == Generalization.CROSS_SESSION and n_sessions is not None and n_sessions < min_sessions:
        findings.append(
            Finding(
                detail=f"Cross-session protocol needs at least {min_sessions} sessions, the source provides {n_sessions}.",
                location="source.session_ids",
            )
        )
    if generalization == Generalization.CROSS_SUBJECT and n_subjects is not None and n_subjects < min_subjects:
        findings.append(
            Finding(
                detail=f"Cross-subject protocol needs at least {min_subjects} subjects, the source provides {n_subjects}.",
                location="source.subject_ids",
            )
        )
    return findings


@methodology_check("few_subjects_for_cross_subject")
def few_subjects_for_cross_subject(ctx: CheckContext) -> list[Finding]:
    """Cross-subject results from very few subjects do not support an uncertainty estimate.

    Params:
        min_subjects: recommended minimum number of independent subjects (default 5).
    """
    if ctx.config.protocol.generalization != Generalization.CROSS_SUBJECT:
        return []

    min_subjects = int(ctx.params.get("min_subjects", 5))
    source = ctx.config.source
    n_subjects = 1 if source.backend == "synthetic" else _known_count(getattr(source, "subject_ids", None))

    if n_subjects is None or n_subjects >= min_subjects:
        return []
    return [
        Finding(
            detail=f"Only {n_subjects} subject(s) selected; per-subject results and their spread are the independent units (recommended >= {min_subjects}).",
            location="source.subject_ids",
        )
    ]


@methodology_check("non_causal_filter_online")
def non_causal_filter_online(ctx: CheckContext) -> list[Finding]:
    """Zero-phase (non-causal) filtering uses future samples, which are unavailable online.

    MNE ``raw.filter`` / ``raw.notch_filter`` use ``phase="zero"`` by default.
    The paradigm band-pass filter is always applied.
    """
    if ctx.config.protocol.usage == Usage.OFFLINE:
        return []

    raw = ctx.config.raw_preprocessing
    filters: list[tuple[str, bool]] = [
        ("raw_preprocessing.high_pass_filter", raw.high_pass_filter.enabled),
        ("raw_preprocessing.low_pass_filter", raw.low_pass_filter.enabled),
        ("raw_preprocessing.notch_filter", raw.notch_filter.enabled),
        ("paradigm.filter", True),
    ]
    return [
        Finding(
            detail=f"Filter '{location}' is applied zero-phase (non-causal) over the whole recording, but usage is declared as '{ctx.config.protocol.usage.value}'.",
            location=location,
        )
        for location, enabled in filters
        if enabled
    ]


@methodology_check("split_reproducibility")
def split_reproducibility(ctx: CheckContext) -> list[Finding]:
    """A shuffled split must have a fixed random seed."""
    split = ctx.config.split
    evaluator = getattr(split, "evaluator", None)

    if evaluator is None:
        # SplitGeneralConfig always requires random_seed.
        return []

    if getattr(evaluator, "shuffle", False) and getattr(evaluator, "random_state", 0) is None:
        return [
            Finding(
                detail=f"Split '{split.backend}' shuffles data but 'random_state' is null; the train/test partition is not reproducible.",
                location="split.evaluator.random_state",
            )
        ]
    return []
