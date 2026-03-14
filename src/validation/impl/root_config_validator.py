import logging
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError

from pipeline.context.run_context import RunContext
from src.types.dto.config.root_config import RootConfig
from src.types.dto.config.experiment_config import ExperimentConfig

class ExperimentConfigValidator():
    def validate(self, config_in: dict[str, Any]) -> ExperimentConfig | None:
        log = logging.getLogger(__name__)
        validation_msgs = []
        try:
            tmp = ExperimentConfig.model_validate(config_in)
            log.info("Config successfully validated")
            return tmp
        except ValidationError as e:
            validation_msgs.append(str(e))
            print(f"validation error: {str(e)}")

        return None