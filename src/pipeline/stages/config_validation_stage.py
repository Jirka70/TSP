from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.config_validator import IConfigValidator
from src.validation.contracts.validation_message import ValidationMessage
from src.validation.contracts.validation_result import ValidationResult


class ConfigValidationStage:
    def __init__(self, validators: list[IConfigValidator]) -> None:
        self._validators = validators

    def run(self, config: ExperimentConfig) -> ValidationResult:
        messages: list[ValidationMessage] = []

        for validator in self._validators:
            result: ValidationResult = validator.validate(config)
            messages.extend(result.messages)

        if any(msg.severity == "error" for msg in messages):
            return ValidationResult(messages=messages)

        return ValidationResult(messages=messages)
