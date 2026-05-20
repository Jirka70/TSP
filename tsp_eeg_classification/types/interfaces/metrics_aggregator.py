from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.types.dto.model.aggregated_metrics_dto import AggregatedMetricsDTO
from tsp_eeg_classification.types.dto.model.training_result_dto import TrainingResultDTO


class IMetricsAggregator(ABC):
    """Contract for aggregating metrics across trained folds."""

    @abstractmethod
    def run(self, result_dto: TrainingResultDTO, run_ctx: RunContext) -> AggregatedMetricsDTO | None:
        """Aggregate the provided training results into summary metrics."""
        raise NotImplementedError
