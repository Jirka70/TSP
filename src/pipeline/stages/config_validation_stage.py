from src.validation.contracts.validation_message import ValidationMessage
from src.validation.contracts.validation_result import ValidationResult


class ConfigValidationStage:
    def __init__(self, validators: list[IConfigValidator]) -> None:
        self._validators = validators

    def run(self, config: ExperimentConfig) -> ValidationResult[ExperimentConfig]:
        messages: list[ValidationMessage] = []

        for validator in self._validators:
            messages.extend(validator.validate(config))

        if any(msg.severity == "error" for msg in messages):
            return ValidationResult(value=None, messages=messages)

        return ValidationResult(value=config, messages=messages)