from abc import ABC, abstractmethod
from typing import Any

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.contracts.validation_result import ValidationResult


class IConfigValidator(ABC):
    @abstractmethod
    def validate(self, config_in: dict[str, Any], config_out: ExperimentConfig) -> ValidationResult:
        raise NotImplementedError
