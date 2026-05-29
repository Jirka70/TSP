from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from src.types.dto.config.astageconfig import AStageConfig


class ResamplingConfig(BaseModel):
    enabled: bool = False
    sfreq: float = Field(gt=0, description="Sampling frequency must be a positive value.")


class HighPassFilterConfig(BaseModel):
    enabled: bool = True
    l_freq: float = Field(ge=0, description="High-pass frequency cannot be negative.")


class LowPassFilterConfig(BaseModel):
    enabled: bool = False
    h_freq: float = Field(gt=0, description="Low-pass frequency must be positive.")


class NotchFilterConfig(BaseModel):
    enabled: bool = True
    freqs: list[float]

    @field_validator("freqs")
    @classmethod
    def check_positive_freqs(cls, v: list[float]) -> list[float]:
        if any(f <= 0 for f in v):
            raise ValueError("All Notch filter frequencies must be greater than 0.")
        return v


class BadChannelsInterpolationConfig(BaseModel):
    enabled: bool = True


class ICAConfig(BaseModel):
    enabled: bool = False
    # n_components can be int (count) or float (variance ratio 0-1)
    n_components: float | int = Field(default=0.95, gt=0)
    method: Literal["infomax", "fastica", "picard"] = "infomax"

    @field_validator("n_components")
    @classmethod
    def validate_ica_components(cls, v: float | int) -> float | int:
        if isinstance(v, float) and v >= 1.0:
            raise ValueError("If n_components is a float, it must be in the range (0, 1) representing variance ratio.")
        return v


class ReReferencingConfig(BaseModel):
    enabled: bool = True
    method: Literal["CSD", "AVERAGE"] = "CSD"


class AnnotateBreakConfig(BaseModel):
    enabled: bool = True
    min_break_duration: float = Field(gt=0, description="Break duration must be positive.")


class RawPreprocessingConfig(AStageConfig):
    backend: Literal["default"]
    resampling: ResamplingConfig
    high_pass_filter: HighPassFilterConfig
    low_pass_filter: LowPassFilterConfig
    notch_filter: NotchFilterConfig
    bad_channels_interpolation: BadChannelsInterpolationConfig
    ica: ICAConfig
    re_referencing: ReReferencingConfig
    annotate_break: AnnotateBreakConfig

    @model_validator(mode="after")
    def validate_filter_hierarchy(self) -> "RawPreprocessingConfig":
        """
        Cross-check logic between different filter settings.
        """
        # Ensure HPF is lower than LPF if both are enabled
        if self.high_pass_filter.enabled and self.low_pass_filter.enabled:
            if self.high_pass_filter.l_freq >= self.low_pass_filter.h_freq:
                raise ValueError(
                    f"High-pass frequency ({self.high_pass_filter.l_freq} Hz) "
                    f"must be lower than Low-pass frequency ({self.low_pass_filter.h_freq} Hz)."
                )
        return self