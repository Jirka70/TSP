from typing import Literal

from pydantic import BaseModel

from src.types.dto.config.astageconfig import AStageConfig


class AlignmentConfig(BaseModel):
    enabled: bool
    tmin_offset: float


class ICAConfig(BaseModel):
    enabled: bool
    n_components: int
    random_state: int
    method: Literal["fastica", "infomax", "picard"]
    eog_threshold: float


class AutoRejectConfig(BaseModel):
    enabled: bool
    n_interpolate: list[int]
    consensus: list[float]
    cv: int


class CSPConfig(BaseModel):
    enabled: bool
    n_components: int
    reg: float | str | None = None
    log: bool
    norm_trace: bool


class EpochPreprocessingConfig(AStageConfig):
    backend: Literal["testing"]
    alignment: AlignmentConfig
    ica: ICAConfig
    autoreject: AutoRejectConfig
    csp: CSPConfig
