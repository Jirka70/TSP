from dataclasses import dataclass

from src.validation.validation_issue import ValidationIssue


@dataclass(frozen=True)
class ValidationResult[T]:
    value: T | None
    messages: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return not any(m.severity == "error" for m in self.messages)