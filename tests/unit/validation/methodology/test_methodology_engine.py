"""Unit tests of the methodology rules engine (DSL, rule model, loader)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.types.dto.config.protocol_config import RigorLevel
from src.validation.methodology.check_registry import registered_checks
from src.validation.methodology.condition_evaluator import MISSING, evaluate_condition, resolve_path
from src.validation.methodology.rule_loader import DEFAULT_RULES_PATH, RulesCatalogError, load_rules_catalog
from src.validation.methodology.rule_model import Condition, MethodologyRule, RuleAction, RulePhase

DATA = {
    "mode": "training",
    "split": {"backend": "general", "ratios": {"test": 0.2}},
    "paradigm": {"window": {"baseline": [-0.5, 0.0]}},
    "evaluation": {"metrics": ["accuracy", "f1_macro"]},
    "flag": True,
}


def _cond(raw: dict) -> Condition:
    return Condition.model_validate(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"path": "split.backend", "eq": "general"}, True),
        ({"path": "split.backend", "ne": "general"}, False),
        ({"path": "split.backend", "in": ["general", "x"]}, True),
        ({"path": "split.backend", "not_in": ["general"]}, False),
        ({"path": "split.ratios.test", "gt": 0.1}, True),
        ({"path": "split.ratios.test", "le": 0.0}, False),
        ({"path": "paradigm.window.baseline.1", "eq": 0.0}, True),
        ({"path": "evaluation.metrics", "contains": "f1_macro"}, True),
        ({"path": "evaluation.metrics", "contains": "balanced_accuracy"}, False),
        ({"path": "flag", "gt": 0}, False),
        ({"path": "split.missing", "exists": False}, True),
        ({"path": "split.missing", "eq": None}, False),
        ({"path": "split.missing", "ne": "x"}, False),
        ({"all": [{"path": "flag", "eq": True}, {"path": "mode", "eq": "training"}]}, True),
        ({"any": [{"path": "flag", "eq": False}, {"path": "mode", "eq": "training"}]}, True),
        ({"not": {"path": "flag", "eq": True}}, False),
    ],
)
def test_condition_evaluation(raw: dict, expected: bool) -> None:
    assert evaluate_condition(_cond(raw), DATA) is expected


def test_resolve_missing_path() -> None:
    assert resolve_path(DATA, "split.ratios.unknown") is MISSING
    assert resolve_path(DATA, "paradigm.window.baseline.5") is MISSING


@pytest.mark.parametrize(
    "raw",
    [
        {"path": "a"},
        {"path": "a", "eq": 1, "ne": 2},
        {"all": [], "path": "a", "eq": 1},
        {"all": [], "any": []},
        {"eq": 1},
    ],
)
def test_invalid_condition_shape_is_rejected(raw: dict) -> None:
    with pytest.raises(ValidationError):
        Condition.model_validate(raw)


def _rule(**kwargs) -> MethodologyRule:
    base = {"id": "X-1", "title": "t", "certainty": "proven", "when": {"path": "flag", "eq": True}}
    base.update(kwargs)
    return MethodologyRule.model_validate(base)


def test_default_severity_by_certainty() -> None:
    proven = _rule()
    suspected = _rule(certainty="suspected")

    assert [proven.action_for(level) for level in RigorLevel] == [RuleAction.WARN, RuleAction.ERROR, RuleAction.ERROR]
    assert [suspected.action_for(level) for level in RigorLevel] == [RuleAction.INFO, RuleAction.WARN, RuleAction.ERROR]


def test_severity_override_keeps_other_defaults() -> None:
    rule = _rule(severity={"standard": "off"})

    assert rule.action_for(RigorLevel.STANDARD) == RuleAction.OFF
    assert rule.action_for(RigorLevel.PUBLICATION) == RuleAction.ERROR


@pytest.mark.parametrize(
    "kwargs",
    [
        {"check": "split_matches_protocol"},  # both when and check
        {"when": None},  # neither
        {"phase": "runtime"},  # declarative runtime rule
        {"params": {"a": 1}},  # params without check
    ],
)
def test_invalid_rule_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        _rule(**kwargs)


def test_default_catalog_loads_and_all_checks_exist() -> None:
    catalog = load_rules_catalog(DEFAULT_RULES_PATH)
    checks = registered_checks()

    assert catalog.rules
    for rule in catalog.rules:
        if rule.check is not None:
            assert rule.check in checks
            assert checks[rule.check].phase == RulePhase.STATIC


def test_catalog_with_unknown_check_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        'version: "t"\nrules:\n  - {id: X-1, title: t, certainty: proven, check: does_not_exist}\n',
        encoding="utf-8",
    )

    with pytest.raises(RulesCatalogError, match="unknown check"):
        load_rules_catalog(path)


def test_catalog_with_duplicate_ids_is_rejected(tmp_path: Path) -> None:
    rule = "  - {id: X-1, title: t, certainty: proven, when: {path: a, eq: 1}}\n"
    path = tmp_path / "rules.yaml"
    path.write_text('version: "t"\nrules:\n' + rule + rule, encoding="utf-8")

    with pytest.raises(RulesCatalogError, match="Duplicate"):
        load_rules_catalog(path)
