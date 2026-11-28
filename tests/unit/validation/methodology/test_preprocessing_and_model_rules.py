"""Rules on preprocessing, its compatibility with ML models and the training run of a loaded model."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.config_validator import ExperimentConfigValidator
from src.validation.methodology.checks.training_run_checks import are_sources_overlapping
from src.validation.methodology.methodology_validator import MethodologyValidator

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = PROJECT_ROOT / "configs"
TRAINING_ML_CONFIG = "test/smoke/synthetic/training_ml"
BENCHMARK_CONFIG = "test/methodology/C_augmentation_before_split"


def load_config(config_name: str, overrides: list[str] | None = None) -> ExperimentConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        hydra_config = compose(config_name=config_name, overrides=overrides or [])
    validation_result = ExperimentConfigValidator().validate(hydra_config)
    assert validation_result.is_valid, [str(message) for message in validation_result.messages]
    return validation_result.config


def find_violated_rule_ids(config: ExperimentConfig) -> set[str]:
    report = MethodologyValidator.from_file().validate(config)
    return {violation.rule.id for violation in report.violations}


def find_violation_locations(config: ExperimentConfig, rule_id: str) -> list[str | None]:
    report = MethodologyValidator.from_file().validate(config)
    return [violation.location for violation in report.violations if violation.rule.id == rule_id]


def load_experiment_config(model_path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    experiment_overrides = [
        "mode=experiment",
        "save_artifacts=experiment",
        "evaluation=experiment",
        f"model_path.path={model_path.as_posix()}",
    ]
    return load_config(TRAINING_ML_CONFIG, experiment_overrides + (overrides or []))


def save_training_run(tmp_path: Path, overrides: list[str] | None = None) -> Path:
    """Write a training run in the artifact saver layout and return the path of its model file."""
    training_config = load_config(TRAINING_ML_CONFIG, overrides)
    run_directory = tmp_path / "riemannian_lda-run"
    (run_directory / "models").mkdir(parents=True)
    (run_directory / "config.json").write_text(json.dumps(training_config.model_dump(mode="json")), encoding="utf-8")
    return run_directory / "models" / "riemannian_lda.joblib"


# --- PRE-003 / PRE-004: covariance rank for Riemannian models -----------


def test_re_referencing_with_unregularized_estimator_is_proven_rank_problem() -> None:
    config = load_config(
        TRAINING_ML_CONFIG,
        ["model.parameters.estimator=scm", "raw_preprocessing.re_referencing.enabled=true"],
    )

    assert "PRE-003" in find_violated_rule_ids(config)


def test_re_referencing_with_shrinkage_estimator_is_accepted() -> None:
    config = load_config(TRAINING_ML_CONFIG, ["raw_preprocessing.re_referencing.enabled=true"])

    violated_rule_ids = find_violated_rule_ids(config)

    assert "PRE-003" not in violated_rule_ids
    assert "PRE-004" not in violated_rule_ids


def test_epoch_ica_with_unregularized_estimator_is_suspected_rank_problem() -> None:
    config = load_config(TRAINING_ML_CONFIG, ["model.parameters.estimator=scm", "epoch_preprocessing.ica.enabled=true"])

    violated_rule_ids = find_violated_rule_ids(config)

    assert "PRE-004" in violated_rule_ids
    assert "PRE-003" not in violated_rule_ids


# --- PRE-006: channel dropout for Riemannian models ------------------------

CHANNEL_DROPOUT_OVERRIDES = ["augmentation=basic", "augmentation.channel_dropout.prob=0.1"]


def test_channel_dropout_with_unregularized_estimator_is_proven_rank_problem() -> None:
    config = load_config(TRAINING_ML_CONFIG, CHANNEL_DROPOUT_OVERRIDES + ["model.parameters.estimator=scm"])

    assert "PRE-006" in find_violated_rule_ids(config)


def test_channel_dropout_with_shrinkage_estimator_is_accepted() -> None:
    config = load_config(TRAINING_ML_CONFIG, CHANNEL_DROPOUT_OVERRIDES + ["model.parameters.estimator=oas"])

    assert "PRE-006" not in find_violated_rule_ids(config)


def test_disabled_channel_dropout_with_unregularized_estimator_is_accepted() -> None:
    config = load_config(
        TRAINING_ML_CONFIG,
        ["augmentation=basic", "augmentation.channel_dropout.prob=0.0", "model.parameters.estimator=scm"],
    )

    assert "PRE-006" not in find_violated_rule_ids(config)


def test_channel_dropout_is_not_checked_in_experiment_mode(tmp_path: Path) -> None:
    model_path = save_training_run(tmp_path)
    config = load_experiment_config(model_path, CHANNEL_DROPOUT_OVERRIDES + ["model.parameters.estimator=scm"])

    assert "PRE-006" not in find_violated_rule_ids(config)


# --- PRE-005: effective band ----------------------------------------------


def test_default_preprocessing_keeps_declared_band() -> None:
    assert "PRE-005" not in find_violated_rule_ids(load_config(TRAINING_ML_CONFIG))


@pytest.mark.parametrize(
    ("overrides", "expected_location"),
    [
        (["raw_preprocessing.high_pass_filter.enabled=true", "raw_preprocessing.high_pass_filter.l_freq=10"],
         "raw_preprocessing.high_pass_filter.l_freq"),
        (["raw_preprocessing.low_pass_filter.enabled=true", "raw_preprocessing.low_pass_filter.h_freq=30"],
         "raw_preprocessing.low_pass_filter.h_freq"),
        (["raw_preprocessing.resampling.enabled=true", "raw_preprocessing.resampling.sfreq=60"],
         "raw_preprocessing.resampling.sfreq"),
        (["paradigm.resampling.sfreq=70"], "paradigm.resampling.sfreq"),
    ],
)
def test_step_narrowing_declared_band_is_reported(overrides: list[str], expected_location: str) -> None:
    config = load_config(TRAINING_ML_CONFIG, overrides)

    assert find_violation_locations(config, "PRE-005") == [expected_location]


# --- EVAL-002: benchmark without fold models --------------------------------


def test_benchmark_without_fold_training_produces_no_estimate() -> None:
    config = load_config(BENCHMARK_CONFIG, ["model.fold_training=false"])

    assert "EVAL-002" in find_violated_rule_ids(config)


def test_benchmark_with_fold_training_is_accepted() -> None:
    config = load_config(BENCHMARK_CONFIG, ["model.fold_training=true"])

    assert "EVAL-002" not in find_violated_rule_ids(config)


# --- REP-003: random forest seed -------------------------------------------


def test_random_forest_config_has_fixed_seed() -> None:
    config = load_config(TRAINING_ML_CONFIG, ["model=machine_learning/riemannian_rf"])

    assert config.model.parameters["random_state"] == 42
    assert "REP-003" not in find_violated_rule_ids(config)


def test_random_forest_without_seed_is_not_reproducible() -> None:
    config = load_config(
        TRAINING_ML_CONFIG,
        ["model=machine_learning/riemannian_rf", "model.parameters.random_state=null"],
    )

    assert "REP-003" in find_violated_rule_ids(config)


# --- LEAK-003: scope of per-recording cleaning ------------------------------


def test_epoch_ica_with_split_inside_recordings_is_reported() -> None:
    config = load_config(TRAINING_ML_CONFIG, ["epoch_preprocessing.ica.enabled=true"])

    assert "LEAK-003" in find_violated_rule_ids(config)


def test_epoch_ica_with_split_between_recordings_is_accepted() -> None:
    config = load_config(BENCHMARK_CONFIG, ["epoch_preprocessing.ica.enabled=true"])

    assert "LEAK-003" not in find_violated_rule_ids(config)


def test_epoch_ica_in_experiment_mode_is_accepted(tmp_path: Path) -> None:
    config = load_experiment_config(tmp_path / "missing-run" / "models" / "model.joblib",
                                    ["epoch_preprocessing.ica.enabled=true"])

    assert "LEAK-003" not in find_violated_rule_ids(config)


# --- Experiment mode vs. training run (LEAK-005, PRE-001, PRE-002) -----------


def test_experiment_on_training_data_is_reported(tmp_path: Path) -> None:
    model_path = save_training_run(tmp_path)
    config = load_experiment_config(model_path)

    assert "LEAK-005" in find_violated_rule_ids(config)


def test_experiment_on_other_synthetic_data_is_accepted(tmp_path: Path) -> None:
    model_path = save_training_run(tmp_path)
    config = load_experiment_config(model_path, ["source.random_seed=7"])

    assert "LEAK-005" not in find_violated_rule_ids(config)


def test_different_model_input_is_proven_mismatch(tmp_path: Path) -> None:
    model_path = save_training_run(tmp_path)
    config = load_experiment_config(model_path, ["source.random_seed=7", "paradigm.filter.fmax=30"])

    assert find_violation_locations(config, "PRE-001") == ["paradigm.filter"]


def test_different_cleaning_is_suspected_mismatch(tmp_path: Path) -> None:
    model_path = save_training_run(tmp_path)
    config = load_experiment_config(model_path, ["source.random_seed=7", "epoch_preprocessing.ica.enabled=true"])

    violated_rule_ids = find_violated_rule_ids(config)

    assert find_violation_locations(config, "PRE-002") == ["epoch_preprocessing.ica"]
    assert "PRE-001" not in violated_rule_ids


def test_parameters_of_block_disabled_in_both_runs_are_ignored(tmp_path: Path) -> None:
    model_path = save_training_run(tmp_path)
    config = load_experiment_config(model_path, ["source.random_seed=7", "epoch_preprocessing.ica.n_components=5"])

    assert "PRE-002" not in find_violated_rule_ids(config)


def test_experiment_without_saved_training_config_reports_nothing(tmp_path: Path) -> None:
    config = load_experiment_config(tmp_path / "missing-run" / "models" / "model.joblib", ["paradigm.filter.fmax=30"])

    violated_rule_ids = find_violated_rule_ids(config)

    assert {"LEAK-005", "PRE-001", "PRE-002"}.isdisjoint(violated_rule_ids)


EXTERNAL_SOURCE = {"backend": "external", "name": "BNCI2014_001", "subject_ids": [1, 2], "session_ids": None,
                   "run_ids": None}


@pytest.mark.parametrize(
    ("experiment_source", "is_overlap_expected"),
    [
        ({**EXTERNAL_SOURCE, "subject_ids": [2, 3]}, True),
        ({**EXTERNAL_SOURCE, "subject_ids": [3]}, False),
        ({**EXTERNAL_SOURCE, "name": "BNCI2014_004"}, False),
        ({**EXTERNAL_SOURCE, "session_ids": ["0"]}, True),
        ({"backend": "synthetic", "random_seed": 42}, False),
    ],
)
def test_external_source_overlap(experiment_source: dict, is_overlap_expected: bool) -> None:
    assert are_sources_overlapping(EXTERNAL_SOURCE, experiment_source) is is_overlap_expected


def test_external_sources_with_disjoint_sessions_do_not_overlap() -> None:
    training_source = {**EXTERNAL_SOURCE, "session_ids": [0]}
    experiment_source = {**EXTERNAL_SOURCE, "session_ids": [1]}

    assert are_sources_overlapping(training_source, experiment_source) is False
