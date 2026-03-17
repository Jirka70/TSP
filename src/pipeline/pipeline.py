from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.experiment_config import ExperimentConfig


class IPipeline(ABC):

    @abstractmethod
    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        raise NotImplementedError
