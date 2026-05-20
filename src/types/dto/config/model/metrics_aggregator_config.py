from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class MetricsAggregatorConfig(AStageConfig):
    """Configuration for aggregating metrics after training."""

    backend: Literal["default"]
