from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class ParadigmConfig(AStageConfig):
    backend: Literal["testing"]
    events: list[str]
    fmin: float
    fmax: float
    tmin: float
    tmax: float
    baseline: list[float]  # Validates the [-0.5, 0.0] range
    resample: float
    reject_by_annotation: bool
    preload: bool
