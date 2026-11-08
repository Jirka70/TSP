"""Validation of the experiment methodology against the rules catalog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.protocol_config import RigorLevel
from src.validation.methodology.check_registry import CheckContext, Finding, get_check
from src.validation.methodology.condition_evaluator import evaluate_condition
from src.validation.methodology.methodology_report import MethodologyReport, RuleViolation, SkippedRule
from src.validation.methodology.rule_loader import DEFAULT_RULES_PATH, load_rules_catalog
from src.validation.methodology.rule_model import MethodologyRule, RuleAction, RulePhase, RulesCatalog
from src.validation.validation_message import ValidationMessageSeverity

log = logging.getLogger(__name__)

_LOG_LEVEL: dict[ValidationMessageSeverity, int] = {
    ValidationMessageSeverity.INFO: logging.INFO,
    ValidationMessageSeverity.WARN: logging.WARNING,
    ValidationMessageSeverity.ERR: logging.ERROR,
}


class MethodologyValidator:
    """Evaluates methodology rules for an experiment.

    The rigor level is taken from ``config.protocol.rigor`` unless overridden.

    Usage::

        validator = MethodologyValidator.from_file()
        report = validator.validate(ex_conf)       # static phase, before the run
        report.save(resolve_report_dir(ex_conf))
        if not report.is_valid: ...                # blocked at this rigor level
    """

    def __init__(self, catalog: RulesCatalog, rigor: RigorLevel | None = None):
        """
        Args:
            catalog: Loaded and validated rules catalog.
            rigor: Optional rigor override (defaults to ``config.protocol.rigor``).
        """
        self._catalog = catalog
        self._rigor_override = rigor

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_RULES_PATH, rigor: RigorLevel | None = None) -> MethodologyValidator:
        """Creates a validator from a rules YAML file (default: ``configs/methodology/rules.yaml``)."""
        return cls(load_rules_catalog(path), rigor)

    @property
    def rules(self) -> list[MethodologyRule]:
        """All rules of the catalog."""
        return list(self._catalog.rules)

    def validate(self, config: ExperimentConfig) -> MethodologyReport:
        """Evaluates all static rules against the configuration (before the run)."""
        return self._run(RulePhase.STATIC, config, data=None)

    def validate_runtime(self, config: ExperimentConfig, data: Any) -> MethodologyReport:
        """Evaluates all runtime rules against runtime data (e.g. ``DatasetSplitDTO``).

        Prepared for runtime guards in pipelines; not wired into pipelines yet.
        """
        return self._run(RulePhase.RUNTIME, config, data=data)

    def _run(self, phase: RulePhase, config: ExperimentConfig, data: Any) -> MethodologyReport:
        rigor = self._rigor_override or config.protocol.rigor
        report = MethodologyReport(phase=phase, rigor=rigor, config=config, rules_version=self._catalog.version)
        dumped = config.model_dump(mode="json")

        for rule in self._catalog.rules:
            skip_reason = self._skip_reason(rule, phase, config, rigor)
            if skip_reason is not None:
                report.skipped_rules.append(SkippedRule(rule_id=rule.id, reason=skip_reason))
                continue

            findings = self._evaluate(rule, config, dumped, data)
            if not findings:
                report.passed_rules.append(rule.id)
                continue

            action = rule.action_for(rigor)
            for finding in findings:
                violation = RuleViolation(rule=rule, action=action, detail=finding.detail, location=finding.location)
                report.violations.append(violation)
                message = violation.to_message()
                log.log(_LOG_LEVEL[message.severity], "Methodology: %s", message)

        return report

    @staticmethod
    def _skip_reason(rule: MethodologyRule, phase: RulePhase, config: ExperimentConfig, rigor: RigorLevel) -> str | None:
        if rule.phase != phase:
            return f"phase '{rule.phase.value}'"
        if not rule.applies_to(config.mode):
            return f"not applicable to mode '{config.mode.value if hasattr(config.mode, 'value') else config.mode}'"
        if rule.action_for(rigor) == RuleAction.OFF:
            return f"off at rigor '{rigor.value}'"
        return None

    @staticmethod
    def _evaluate(rule: MethodologyRule, config: ExperimentConfig, dumped: dict[str, Any], data: Any) -> list[Finding]:
        if rule.when is not None:
            if evaluate_condition(rule.when, dumped):
                return [Finding(detail=rule.description or rule.title)]
            return []

        assert rule.check is not None  # guaranteed by MethodologyRule.validate_detection
        check = get_check(rule.check)
        return check.fn(CheckContext(config=config, params=rule.params, data=data))
