"""Root config validator."""

import logging
from typing import Any

from pydantic import ValidationError

from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.validation_message import ValidationMessage
from validation.validation_message import ValidationResult


class ExperimentConfigValidator:
    """Main (root) config validator."""

    def validate(self, config_in: dict[str, Any]) -> ValidationResult:
        """Validates a raw configuration dictionary and compiles any schema errors.

        Attempts to parse the input dictionary into an ExperimentConfig model.
        If the validation fails, the errors are caught, logged, and bundled
        into the returned result object.

        Args:
            config_in (dict[str, Any]): The raw experiment configuration loaded
            by hydra.

        Returns:
            ValidationResult: An object containing the successfully parsed
                `ExperimentConfig` (if valid, otherwise None) and a list of
                `ValidationMessage` objects representing any errors encountered.
        """
        log = logging.getLogger(__name__)

        validation_msgs = []
        ex_conf: ExperimentConfig | None = None

        try:
            ex_conf = ExperimentConfig.model_validate(config_in)
        except ValidationError as e:
            for msg in e.errors():
                validation_msg = ValidationMessage(log=msg)
                log.error(validation_msg)
                validation_msgs.append(validation_msg)

        return ValidationResult(messages=validation_msgs, config=ex_conf)
