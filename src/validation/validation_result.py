from dataclasses import dataclass
from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.validation_message import ValidationMessage, ValidationMessageSeverity


@dataclass(frozen=True)
class ValidationResult:
    messages: list[ValidationMessage]
    config: ExperimentConfig

    @property
    def is_valid(self) -> bool:
        return not any(m.severity == ValidationMessageSeverity.ERR for m in self.messages)
