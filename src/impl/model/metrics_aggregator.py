import logging
from collections import defaultdict

import numpy as np

from src.pipeline.context.run_context import RunContext
from src.types.dto.model.aggregated_metrics_dto import AggregatedMetricsDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.interfaces.metrics_aggregator import IMetricsAggregator

log = logging.getLogger(__name__)


class MetricsAggregator(IMetricsAggregator):
    """
    Generický evaluátor křížové validace.
    Umí počítat globální statistiky i statistiky seskupené podle metadat (např. subject_id).
    """

    def __init__(self, group_by_key: str | None = "subject_id") -> None:
        self._group_by_key = group_by_key

    def run(self, result_dto: TrainingResultDTO, run_ctx: RunContext) -> AggregatedMetricsDTO:
        if not result_dto.trained_models:
            raise ValueError("Žádné modely k evaluaci.")

        first_model = result_dto.trained_models[0]
        model_name = first_model.model_name
        metric_name = first_model.best_validation_metric_name or "accuracy"

        all_values: list[float] = []
        grouped_values: dict[str, list[float]] = defaultdict(list)
        fold_results: dict[int, float] = {}

        for trained_model in result_dto.trained_models:
            val = self._extract_metric(trained_model, metric_name)
            fold_idx = trained_model.fold_idx if trained_model.fold_idx is not None else len(all_values)

            all_values.append(val)
            fold_results[fold_idx] = val

            if self._group_by_key:
                group_id = str(trained_model.metadata.get(self._group_by_key, "unknown"))
                grouped_values[group_id].append(val)

        global_mean = float(np.mean(all_values))
        global_std = float(np.std(all_values))

        log.info(f"\n=== Globální výsledky: {model_name} ===")
        log.info(f"Mean: {global_mean:.4f} ± {global_std:.4f}")

        if self._group_by_key and grouped_values:
            log.info(f"\n--- Výsledky podle: {self._group_by_key} ---")
            for group_id, values in grouped_values.items():
                g_mean = float(np.mean(values))
                g_std = float(np.std(values))
                log.info(f"[{self._group_by_key}: {group_id}] -> Mean: {g_mean:.4f} ± {g_std:.4f} (Foldu: {len(values)})")

        return AggregatedMetricsDTO(
            model_name=model_name,
            metric_name=metric_name,
            mean=global_mean,
            std=global_std,
            fold_results=fold_results,
        )

    def _extract_metric(self, trained_model, metric_name: str) -> float:
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
