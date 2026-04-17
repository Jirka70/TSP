import logging
from collections import defaultdict

import numpy as np

from src.types.dto.model.aggregated_metrics_dto import AggregatedMetricsDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.interfaces.metrics_aggregator import IMetricsAggregator

log = logging.getLogger(__name__)


class MetricsAggregator(IMetricsAggregator):
    """
    Generický evaluátor křížové validace.
    Umí počítat globální statistiky i statistiky seskupené podle metadat (např. subject_id).
    """

    def run(self, result_dto: TrainingResultDTO, group_by_key: str | None = "subject_id") -> AggregatedMetricsDTO:
        if not result_dto.trained_models:
            raise ValueError("Žádné modely k evaluaci.")

        first_model = result_dto.trained_models[0]
        model_name = first_model.model_name
        metric_name = first_model.best_validation_metric_name or "accuracy"

        # Všechny hodnoty pro globální průměr
        all_values: list[float] = []

        # Slovník pro seskupení (např. { 1: [0.88, 0.86, ...], 2: [0.54, 0.55, ...] })
        grouped_values = defaultdict(list)
        fold_results: dict[int, float] = {}

        for trained_model in result_dto.trained_models:
            val = trained_model.best_validation_metric_value or 0.0

            all_values.append(val)
            fold_results[trained_model.fold_idx or len(all_values)] = val

            # Seskupení podle zadaného klíče (např. subject_id) z metadat
            if group_by_key:
                group_id = trained_model.metadata.get(group_by_key, "unknown")
                grouped_values[group_id].append(val)

        # 1. Globální statistiky
        global_mean = float(np.mean(all_values))
        global_std = float(np.std(all_values))

        log.info(f"\n=== Globální výsledky: {model_name} ===")
        log.info(f"Mean: {global_mean:.4f} ± {global_std:.4f}")

        # 2. Seskupené statistiky (Subject-wise)
        grouped_stats = {}
        if group_by_key and grouped_values:
            log.info(f"\n--- Výsledky podle: {group_by_key} ---")
            for group_id, values in grouped_values.items():
                g_mean = float(np.mean(values))
                g_std = float(np.std(values))
                grouped_stats[str(group_id)] = {"mean": g_mean, "std": g_std}
                log.info(f"[{group_by_key}: {group_id}] -> Mean: {g_mean:.4f} ± {g_std:.4f} (Foldů: {len(values)})")

        return AggregatedMetricsDTO(
            model_name=model_name,
            metric_name=metric_name,
            mean=global_mean,
            std=global_std,
            fold_results=fold_results,
            # (pozn. do AggregatedMetricsDTO si můžeš přidat dict na ty grouped_stats)
        )
