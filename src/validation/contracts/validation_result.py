from dataclasses import dataclass

from src.validation.contracts.validation_message import ValidationMessage


@dataclass(frozen=True)
class ValidationResult:
    messages: list[ValidationMessage]

    @property
    def is_valid(self) -> bool:
        return not any(m.severity == "error" for m in self.messages)
