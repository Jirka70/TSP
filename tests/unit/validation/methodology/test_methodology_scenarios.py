"""Demonstration scenarios A-F from the methodology document (section 10)."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.protocol_config import RigorLevel
from src.validation.config_validator import ExperimentConfigValidator
from src.validation.methodology.methodology_validator import MethodologyValidator
from src.validation.methodology.rule_model import RuleAction

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load(config_name: str, overrides: list[str] | None = None) -> ExperimentConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    result = ExperimentConfigValidator().validate(cfg)
    assert result.is_valid, [str(m) for m in result.messages]
    return result.config


def _actions(config: ExperimentConfig, rigor: RigorLevel | None = None) -> dict[str, RuleAction]:
    report = MethodologyValidator.from_file(rigor=rigor).validate(config)
    return {v.rule.id: v.action for v in report.violations}


@pytest.mark.parametrize(
    ("scenario", "rule_id", "action_at_standard"),
    [
        ("A_csp_before_split", "LEAK-001", RuleAction.ERROR),
        ("B_selection_on_test", "SEL-001", RuleAction.ERROR),
        ("C_augmentation_before_split", "LEAK-002", RuleAction.ERROR),
        ("D_random_split_cross_subject", "PROT-001", RuleAction.ERROR),
        ("E_non_causal_online", "ONL-001", RuleAction.WARN),
    ],
)
def test_faulty_scenario_is_detected(scenario: str, rule_id: str, action_at_standard: RuleAction) -> None:
    config = _load(f"test/methodology/{scenario}")

    actions = _actions(config)

    assert actions.get(rule_id) == action_at_standard


@pytest.mark.parametrize(
    "scenario",
    ["A_csp_before_split", "B_selection_on_test", "C_augmentation_before_split", "D_random_split_cross_subject"],
)
def test_blocking_scenarios_block_at_standard_but_not_at_exploratory(scenario: str) -> None:
    config = _load(f"test/methodology/{scenario}")

    assert not MethodologyValidator.from_file().validate(config).is_valid
    assert MethodologyValidator.from_file(rigor=RigorLevel.EXPLORATORY).validate(config).is_valid


def test_online_scenario_warns_at_standard_and_blocks_at_publication() -> None:
    config = _load("test/methodology/E_non_causal_online")

    assert MethodologyValidator.from_file().validate(config).is_valid
    assert not MethodologyValidator.from_file(rigor=RigorLevel.PUBLICATION).validate(config).is_valid


@pytest.mark.parametrize("rigor", list(RigorLevel))
def test_correct_cross_session_scenario_has_no_violation(rigor: RigorLevel) -> None:
    config = _load("test/methodology/F_correct_cross_session")

    report = MethodologyValidator.from_file(rigor=rigor).validate(config)

    assert report.violations == []
    assert report.is_valid


def test_rigor_is_taken_from_protocol() -> None:
    config = _load("test/methodology/A_csp_before_split", ["protocol.rigor=exploratory"])

    report = MethodologyValidator.from_file().validate(config)

    assert report.rigor == RigorLevel.EXPLORATORY
    assert report.is_valid


def test_report_is_saved(tmp_path: Path) -> None:
    config = _load("test/methodology/A_csp_before_split")
    report = MethodologyValidator.from_file().validate(config)

    path = report.save(tmp_path)

    content = path.read_text(encoding="utf-8")
    assert '"LEAK-001"' in content
    assert '"is_valid": false' in content


@pytest.mark.parametrize(
    "config_name",
    [
        "test/smoke/synthetic/training_ml",
        "test/smoke/synthetic/training_dl_fold",
        "test/smoke/synthetic/training_dl_final_only",
    ],
)
def test_ci_smoke_configs_are_not_blocked(config_name: str) -> None:
    config = _load(config_name)

    assert MethodologyValidator.from_file().validate(config).is_valid
