from typing import Literal

from pydantic import BaseModel

from src.types.dto.config.astageconfig import AStageConfig


class EvaluationConfig(AStageConfig):
    backend: Literal["default"]
    metrics: list[str]