from abc import ABC, abstractmethod

from src.types.dto.model.aggregated_metrics_dto import AggregatedMetricsDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO


class IMetricsAggregator(ABC):
    @abstractmethod
    def run(self, result_dto: TrainingResultDTO, group_by_key: str | None = "subject_id") -> AggregatedMetricsDTO:
        raise NotImplementedError
