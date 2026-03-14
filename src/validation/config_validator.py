import logging
from typing import Any

from pydantic import ValidationError

from src.types.dto.config.experiment_config import ExperimentConfig
from validation.validation_message import ValidationMessage
from validation.validation_result import ValidationResult


class ExperimentConfigValidator():
    def validate(self, config_in: dict[str, Any]) -> ValidationResult:
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