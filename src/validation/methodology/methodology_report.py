"""Result of the methodology validation and its serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.protocol_config import RigorLevel
from src.validation.methodology.rule_model import MethodologyRule, RuleAction, RulePhase
from src.validation.validation_message import ValidationMessage, ValidationMessageSeverity

REPORT_FILE_NAME = "methodology_report.json"

ACTION_TO_SEVERITY: dict[RuleAction, ValidationMessageSeverity] = {
    RuleAction.INFO: ValidationMessageSeverity.INFO,
    RuleAction.WARN: ValidationMessageSeverity.WARN,
    RuleAction.ERROR: ValidationMessageSeverity.ERR,
}


@dataclass(frozen=True)
class RuleViolation:
    """A violated rule together with the resolved action and finding."""

    rule: MethodologyRule
    action: RuleAction
    detail: str
    location: str | None = None

    def to_message(self) -> ValidationMessage:
        """Converts the violation into a generic validation message."""
        return ValidationMessage.from_rule(
            rule_id=self.rule.id,
            message=f"{self.rule.title}: {self.detail}" if self.detail != self.rule.title else self.rule.title,
            severity=ACTION_TO_SEVERITY[self.action],
            location=self.location,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "rule_id": self.rule.id,
            "title": self.rule.title,
            "action": self.action.value,
            "certainty": self.rule.certainty.value,
            "phase": self.rule.phase.value,
            "section": self.rule.section,
            "scenario": self.rule.scenario,
            "detail": self.detail,
            "location": self.location,
            "impact": self.rule.impact,
            "remediation": self.rule.remediation,
        }


@dataclass(frozen=True)
class SkippedRule:
    """A rule that was not evaluated, with the reason."""

    rule_id: str
    reason: str


@dataclass
class MethodologyReport:
    """Outcome of one methodology validation run (static or runtime phase)."""

    phase: RulePhase
    rigor: RigorLevel
    config: ExperimentConfig
    rules_version: str
    violations: list[RuleViolation] = field(default_factory=list)
    passed_rules: list[str] = field(default_factory=list)
    skipped_rules: list[SkippedRule] = field(default_factory=list)
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def messages(self) -> list[ValidationMessage]:
        """Violations as generic validation messages."""
        return [violation.to_message() for violation in self.violations]

    @property
    def is_valid(self) -> bool:
        """``False`` when any violated rule blocks the experiment at the current rigor."""
        return not any(violation.action == RuleAction.ERROR for violation in self.violations)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (stored next to the experiment outputs)."""
        return {
            "created_at_utc": self.created_at_utc.isoformat(),
            "phase": self.phase.value,
            "rigor": self.rigor.value,
            "rules_version": self.rules_version,
            "is_valid": self.is_valid,
            "mode": self.config.mode.value if hasattr(self.config.mode, "value") else str(self.config.mode),
            "protocol": self.config.protocol.model_dump(mode="json"),
            "summary": {action.value: sum(1 for v in self.violations if v.action == action) for action in ACTION_TO_SEVERITY},
            "violations": [violation.to_dict() for violation in self.violations],
            "passed_rules": self.passed_rules,
            "skipped_rules": [{"rule_id": s.rule_id, "reason": s.reason} for s in self.skipped_rules],
        }

    def save(self, output_dir: Path | str) -> Path:
        """Writes the report as JSON into ``output_dir`` and returns the file path."""
        path = Path(output_dir) / REPORT_FILE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as report_file:
            json.dump(self.to_dict(), report_file, indent=2, ensure_ascii=False)
        return path


def resolve_report_dir(config: ExperimentConfig) -> Path:
    """Hydra run output directory when available, otherwise ``config.output_dir``."""
    try:
        from hydra.core.hydra_config import HydraConfig  # pylint: disable=import-outside-toplevel

        return Path(HydraConfig.get().runtime.output_dir)
    except (ValueError, KeyError, RuntimeError, ImportError):
        return Path(config.output_dir)
