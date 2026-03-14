from enum import Enum
from typing import Any


class ValidationMessageSeverity(Enum):
    INFO = "info"
    WARN = "warning"
    ERR = "error"


class ValidationMessage:
    code: str
    message: str
    location: str | None = None
    severity: ValidationMessageSeverity = ValidationMessageSeverity.INFO

    def __init__(self, log: dict[str, Any]):
        self.code = log["type"]
        self.message = log["msg"]
        self.loc = ".".join(log["loc"])
        self.severity = ValidationMessageSeverity.ERR

    def __str__(self):
        return f"{self.loc}: {self.message}"