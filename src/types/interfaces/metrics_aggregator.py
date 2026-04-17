from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.model.aggregated_metrics_dto import AggregatedMetricsDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO


class IMetricsAggregator(ABC):
    @abstractmethod
    def run(self, result_dto: TrainingResultDTO, run_ctx: RunContext) -> AggregatedMetricsDTO:
        raise NotImplementedError
