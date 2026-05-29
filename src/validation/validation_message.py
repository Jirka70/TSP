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

    def __init__(self, log: dict[str, Any]):
        """Initialize the ValidationMessage object.

        Args:
            log (dict[str, Any]): result of pydantic validation.
        """
        self.code = log["type"]
        self.message = log["msg"]
        self.loc = ".".join(log["loc"])
        self.severity = ValidationMessageSeverity.ERR

    def __str__(self):
        """String representation of the ValidationMessage object."""
        return f"{self.loc}: {self.message}"


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
