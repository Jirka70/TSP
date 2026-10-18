from typing import Literal
from pydantic import BaseModel, Field, field_validator
from src.types.dto.config.astageconfig import AStageConfig

class AutoRejectConfig(BaseModel):
    enabled: bool
    n_interpolate: list[int]
    consensus: list[float]
    # Cross-validation requires at least 2 folds
    cv: int = Field(ge=2)
    random_state: int = 42

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

class CSDConfig(BaseModel):
    enabled: bool = False

class CSPConfig(BaseModel):
    enabled: bool
    n_components: int = Field(gt=0)
    reg: float | str | None = None
    log: bool
    norm_trace: bool

class EpochPreprocessingConfig(AStageConfig):
    backend: Literal["default"]
    autoreject: AutoRejectConfig
    csd: CSDConfig = Field(default_factory=CSDConfig)
    csp: CSPConfig