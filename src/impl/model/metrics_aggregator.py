import logging
from collections import defaultdict

import numpy as np

from src.pipeline.context.run_context import RunContext
from src.types.dto.model.aggregated_metrics_dto import AggregatedMetricsDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.interfaces.metrics_aggregator import IMetricsAggregator
from src.types.dto.model.trained_model_dto import TrainedModelDTO

log = logging.getLogger(__name__)


class MetricsAggregator(IMetricsAggregator):
    """Aggregate fold-level metrics into global and grouped summaries."""

    def __init__(self, group_by_key: str | None = "subject_id") -> None:
        """
        Initialize the aggregator with an optional metadata grouping key.

        Args:
            group_by_key (Optional[str]): The metadata key used to group the results
                (e.g., 'subject_id' to evaluate per-subject performance).
        """
        self._group_by_key = group_by_key

    def run(self, result_dto: TrainingResultDTO, run_ctx: RunContext) -> AggregatedMetricsDTO | None:
        """
        Aggregate metrics across trained folds and optionally group them by metadata.

        Args:
            result_dto (TrainingResultDTO): DTO containing models and metrics from all trained folds.
            run_ctx (RunContext): Context of the current execution run.

        Returns:
            Optional[AggregatedMetricsDTO]: DTO containing global and fold-level statistics,
                or None if no trained models were provided.
        """
        if not result_dto.trained_models or len(result_dto.trained_models) == 0:
            log.warning("No trained models were provided for aggregation.")
            return None

        first_model : TrainedModelDTO = result_dto.trained_models[0]
        model_name : str = first_model.model_name
        metric_name : str = first_model.best_validation_metric_name or "accuracy"

        all_values: list[float] = []
        grouped_values: dict[str, list[float]] = defaultdict(list)
        fold_results: dict[int, float] = {}

        for trained_model in result_dto.trained_models:
            val : float = self._extract_metric(trained_model, metric_name)

            # Determine fold index fallback if not explicitly provided
            fold_idx : int = (
                trained_model.fold_idx
                if trained_model.fold_idx is not None
                else len(all_values)
            )

            all_values.append(val)
            fold_results[fold_idx] = val

            if self._group_by_key:
                group_id : str = str(trained_model.metadata.get(self._group_by_key, "unknown"))
                grouped_values[group_id].append(val)

        global_mean : float = float(np.mean(all_values))
        global_std : float = float(np.std(all_values))

        log.info(f"\n=== Global results: {model_name} ===")
        log.info(f"Mean: {global_mean:.4f} ± {global_std:.4f}")

        if self._group_by_key and grouped_values:
            log.info(f"\n--- Results grouped by: {self._group_by_key} ---")
            for group_id, values in grouped_values.items():
                g_mean : float = float(np.mean(values))
                g_std : float = float(np.std(values))
                log.info(f"[{self._group_by_key}: {group_id}] -> Mean: {g_mean:.4f} ± {g_std:.4f} (Folds: {len(values)})")

        return AggregatedMetricsDTO(
            model_name=model_name,
            metric_name=metric_name,
            mean=global_mean,
            std=global_std,
            fold_results=fold_results,
        )

    def _extract_metric(self, trained_model, metric_name: str) -> float:
        """
        Extract the most suitable metric value from a trained model DTO.

        It looks for precalculated best validation values, then falls back to the
        last logged validation metric, and finally to the last training metric.

        Args:
            trained_model (TrainedModelDTO): The trained model structure containing metrics history.
            metric_name (str): Name of the metric to look for.

        Returns:
            float: Extracted metric value, or 0.0 if no matching metric is found.
        """
        if trained_model.best_validation_metric_value is not None:
            return float(trained_model.best_validation_metric_value)

        if trained_model.history:
            val_metrics = trained_model.history.validation_metrics.get(metric_name)
            if val_metrics:
                return float(val_metrics[-1])

            train_metrics = trained_model.history.train_metrics.get(metric_name)
            if train_metrics:
                return float(train_metrics[-1])

        return 0.0
