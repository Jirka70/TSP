from typing import Literal

from pydantic import BaseModel

from src.types.dto.config.astageconfig import AStageConfig


class ICAConfig(BaseModel):
    n_components: int
    random_state: int
    method: Literal["fastica", "infomax", "picard"]
    eog_threshold: float


class CSPConfig(BaseModel):
    n_components: int
    reg: float | str | None = None
    log: bool
    norm_trace: bool


class EpochPreprocessingConfig(AStageConfig):
    backend: Literal["testing"]
    ica: ICAConfig
    csp: CSPConfig
