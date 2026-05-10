from typing import Literal, Union, Optional
from pydantic import BaseModel, Field, model_validator

from src.types.dto.config.astageconfig import AStageConfig


class BandpassFilterConfig(BaseModel):
    fmin: float = 8.0
    fmax: float = 35.0

    @model_validator(mode='after')
    def validate_frequencies(self) -> 'BandpassFilterConfig':
        if self.fmin < 0 or self.fmax <= 0:
            raise ValueError(f"Frequencies must be positive. Given fmin={self.fmin}, fmax={self.fmax}")
        if self.fmin >= self.fmax:
            raise ValueError(f"Lower frequency fmin ({self.fmin}) must be less than fmax ({self.fmax}).")
        return self


class EpochWindowConfig(BaseModel):
    tmin: float = -0.5
    tmax: float = 4.0
    baseline: Optional[tuple[float, float]] = (-0.5, 0.0)

    @model_validator(mode='after')
    def validate_window(self) -> 'EpochWindowConfig':
        if self.tmin >= self.tmax:
            raise ValueError(f"Epoch start tmin ({self.tmin}) must be less than tmax ({self.tmax}).")

        if self.baseline:
            b_min, b_max = self.baseline
            if b_min >= b_max:
                raise ValueError(f"Baseline start ({b_min}) must be less than end ({b_max}).")
            if b_min < self.tmin or b_max > self.tmax:
                raise ValueError(f"Baseline {self.baseline} exceeds epoch boundaries [{self.tmin}, {self.tmax}].")
        return self


class EpochResamplingConfig(BaseModel):
    enabled: bool = False
    sfreq: float = 128.0

    @model_validator(mode='after')
    def validate_sfreq(self) -> 'EpochResamplingConfig':
        if self.enabled and self.sfreq <= 0:
            raise ValueError(f"Sampling frequency must be greater than 0. Given: {self.sfreq}")
        return self


class ParadigmConfig(AStageConfig):
    implementation: Literal["custom", "moabb"] = "custom"
    backend: Literal["testing"] = "testing"

    events: Union[dict[str, int], list[str]]

    filter: BandpassFilterConfig = Field(default_factory=BandpassFilterConfig)
    window: EpochWindowConfig = Field(default_factory=EpochWindowConfig)
    resampling: EpochResamplingConfig = Field(default_factory=EpochResamplingConfig)

    reject_by_annotation: bool = True
    preload: bool = True

    @model_validator(mode='after')
    def validate_events(self) -> 'ParadigmConfig':
        if not self.events:
            raise ValueError("'events' configuration must not be empty. Specify at least one event.")
        return self