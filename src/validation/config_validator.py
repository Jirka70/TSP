from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.contracts.validation_result import ValidationResult


class IConfigValidator(ABC):
    @abstractmethod
    def validate(self, config: ExperimentConfig, run_ctx: RunContext) -> ValidationResult:
        raise NotImplementedError
