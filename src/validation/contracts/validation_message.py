from dataclasses import dataclass
from enum import Enum


class ValidationMessageSeverity(Enum):
    INFO = "info"
    WARN = "warning"
    ERR = "error"


@dataclass(slots=True, frozen=True)
class ValidationMessage:
    code: str
    message: str
    location: str | None = None
    severity: ValidationMessageSeverity = ValidationMessageSeverity.INFO
