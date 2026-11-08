"""Validation utility module."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.types.dto.config.experiment_config import ExperimentConfig


class ValidationMessageSeverity(Enum):
    """Severity enum for validation messages."""

    INFO = "info"
    WARN = "warning"
    ERR = "error"


class ValidationMessage:
    """Validation message for validation results."""

    code: str
    message: str
    location: str | None = None
    severity: ValidationMessageSeverity = ValidationMessageSeverity.INFO
    rule_id: str | None = None

    def __init__(self, log: dict[str, Any]):
        """Initialize the ValidationMessage object.

        Args:
            log (dict[str, Any]): result of pydantic validation.
        """
        self.code = log["type"]
        self.message = log["msg"]
        self.loc = ".".join(str(part) for part in log["loc"])
        self.location = self.loc
        self.severity = ValidationMessageSeverity.ERR

    @classmethod
    def from_rule(
        cls,
        rule_id: str,
        message: str,
        severity: ValidationMessageSeverity,
        location: str | None = None,
    ) -> "ValidationMessage":
        """Creates a message for a violated methodology rule.

        Args:
            rule_id: Id of the violated rule (also used as ``code``).
            message: Human-readable description of the violation.
            severity: Severity resolved from the rule action and rigor level.
            location: Config path the violation relates to.
        """
        msg = cls.__new__(cls)
        msg.code = rule_id
        msg.rule_id = rule_id
        msg.message = message
        msg.loc = location or ""
        msg.location = location
        msg.severity = severity
        return msg

    def __str__(self):
        """String representation of the ValidationMessage object."""
        prefix = f"[{self.rule_id}] " if self.rule_id else ""
        location = f"{self.loc}: " if self.loc else ""
        return f"{prefix}{location}{self.message}"


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation run."""

    messages: list[ValidationMessage]
    config: ExperimentConfig

    @property
    def is_valid(self) -> bool:
        """Whether the result is valid."""
        return not any(
            m.severity == ValidationMessageSeverity.ERR for m in self.messages
        )
