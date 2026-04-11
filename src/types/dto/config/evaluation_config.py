from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class EvaluationConfig(AStageConfig):
    backend: Literal["default"]
    metrics: list[str]


class SklearnEvaluationConfig(AStageConfig):
    backend: Literal["sklearn"]
    metrics: list[str]
