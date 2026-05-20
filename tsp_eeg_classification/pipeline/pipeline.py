from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.types.dto.config.experiment_config import ExperimentConfig


class IPipeline(ABC):
    @abstractmethod
    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        raise NotImplementedError
