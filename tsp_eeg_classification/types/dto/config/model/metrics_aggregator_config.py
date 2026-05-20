from typing import Literal

from tsp_eeg_classification.types.dto.config.astageconfig import AStageConfig


class MetricsAggregatorConfig(AStageConfig):
    """Configuration for aggregating metrics after training."""

    backend: Literal["default"]
