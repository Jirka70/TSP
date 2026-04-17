from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class MetricsAggregatorConfig(AStageConfig):
    backend: Literal["default"]
