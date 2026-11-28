"""Rules LEAK-002 and LEAK-006 on raw augmentation before the split."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.config_validator import ExperimentConfigValidator
from src.validation.methodology.methodology_validator import MethodologyValidator

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = PROJECT_ROOT / "configs"
RAW_AUGMENTATION_CONFIG = "test/methodology/C_augmentation_before_split"


def load_config(config_name: str, overrides: list[str] | None = None) -> ExperimentConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        hydra_config = compose(config_name=config_name, overrides=overrides or [])
    validation_result = ExperimentConfigValidator().validate(hydra_config)
    assert validation_result.is_valid, [str(message) for message in validation_result.messages]
    return validation_result.config


def find_violated_rule_ids(config: ExperimentConfig) -> set[str]:
    report = MethodologyValidator.from_file().validate(config)
    return {violation.rule.id for violation in report.violations}


@pytest.mark.parametrize(
    "split_name",
    ["training/general", "cross_validation/moabb_cross_session", "cross_validation/moabb_within_subject"],
)
def test_split_mixing_copy_and_original_reports_leak_and_augmented_evaluation(split_name: str) -> None:
    config = load_config(RAW_AUGMENTATION_CONFIG, [f"split={split_name}"])

    violated_rule_ids = find_violated_rule_ids(config)

    assert {"LEAK-002", "LEAK-006"} <= violated_rule_ids


@pytest.mark.parametrize(
    "split_name",
    ["cross_validation/moabb_cross_subject", "cross_validation/moabb_within_session"],
)
def test_split_keeping_copy_and_original_apart_reports_only_augmented_evaluation(split_name: str) -> None:
    config = load_config(RAW_AUGMENTATION_CONFIG, [f"split={split_name}"])

    violated_rule_ids = find_violated_rule_ids(config)

    assert "LEAK-002" not in violated_rule_ids
    assert "LEAK-006" in violated_rule_ids


def test_disabled_raw_augmentation_reports_no_raw_augmentation_rule() -> None:
    config = load_config(RAW_AUGMENTATION_CONFIG, ["raw_augmentation=none"])

    violated_rule_ids = find_violated_rule_ids(config)

    assert violated_rule_ids.isdisjoint({"LEAK-002", "LEAK-006"})
