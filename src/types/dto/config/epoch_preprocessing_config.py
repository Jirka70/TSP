from typing import Literal
from pydantic import BaseModel, Field, field_validator
from src.types.dto.config.astageconfig import AStageConfig

class AlignmentConfig(BaseModel):
    enabled: bool
    # Time shift usually stays within reasonable bounds of the epoch length
    tmin_offset: float = Field(default=0.0, ge=-10.0, le=10.0)

class ICAConfig(BaseModel):
    enabled: bool
    # Number of components must be positive and typically doesn't exceed channel count
    n_components: int = Field(gt=0, le=100)
    random_state: int
    method: Literal["fastica", "infomax", "picard"]
    # EOG threshold is a Z-score multiplier, must be positive
    eog_threshold: float = Field(gt=0)

class AutoRejectConfig(BaseModel):
    enabled: bool
    n_interpolate: list[int]
    consensus: list[float]
    # Cross-validation requires at least 2 folds
    cv: int = Field(ge=2)

    @field_validator('n_interpolate')
    @classmethod
    def check_n_interpolate(cls, v):
        if not v:
            raise ValueError("n_interpolate list cannot be empty")
        if any(x < 0 for x in v):
            raise ValueError("n_interpolate values must be non-negative")
        return v

    @field_validator('consensus')
    @classmethod
    def check_consensus(cls, v):
        if not v:
            raise ValueError("consensus list cannot be empty")
        # Consensus values represent a fraction of channels (0.0 to 1.0)
        if any(not (0 <= x <= 1) for x in v):
            raise ValueError("consensus values must be between 0.0 and 1.0")
        return v

class CSPConfig(BaseModel):
    enabled: bool
    n_components: int = Field(gt=0)
    reg: float | str | None = None
    log: bool
    norm_trace: bool

class EpochPreprocessingConfig(AStageConfig):
    backend: Literal["testing"]
    alignment: AlignmentConfig
    ica: ICAConfig
    autoreject: AutoRejectConfig
    csp: CSPConfig