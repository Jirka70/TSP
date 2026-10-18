from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from src.types.dto.config.astageconfig import AStageConfig


class ChannelSelectionConfig(BaseModel):
    # Only neurophysiological channel types are allowed. "stim" is intentionally excluded:
    # the stimulus channel encodes class labels and would leak them into the model input.
    picks: list[Literal["eeg", "eog"]] = Field(default_factory=lambda: ["eeg", "eog"], min_length=1)


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
    # "infomax" and "picard" are run in their extended variants (sub- and super-Gaussian sources)
    method: Literal["infomax", "fastica", "picard"] = "infomax"
    random_state: int = 42
    # Z-score threshold for flagging EOG-correlated components (requires EOG channels)
    eog_threshold: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def check_eog_threshold_reachable(self):
        # The z-score of one component among n can never exceed sqrt(n - 1)
        if isinstance(self.n_components, int) and self.eog_threshold >= (self.n_components - 1) ** 0.5:
            raise ValueError(
                f"eog_threshold={self.eog_threshold} can never be reached with n_components={self.n_components} "
                f"(max z-score is {(self.n_components - 1) ** 0.5:.2f}). Lower the threshold or add components."
            )
        return self

    @field_validator("n_components")
    @classmethod
    def validate_ica_components(cls, v: float | int) -> float | int:
        if isinstance(v, float) and v >= 1.0:
            raise ValueError("If n_components is a float, it must be in the range (0, 1) representing variance ratio.")
        return v


class ReReferencingConfig(BaseModel):
    enabled: bool = True
    # CSD lives in epoch preprocessing (after AutoReject), since it changes channel type to "csd"
    method: Literal["AVERAGE"] = "AVERAGE"


class AnnotateBreakConfig(BaseModel):
    enabled: bool = True
    min_break_duration: float = Field(gt=0, description="Break duration must be positive.")


class RawPreprocessingConfig(AStageConfig):
    backend: Literal["default"]
    channel_selection: ChannelSelectionConfig = Field(default_factory=ChannelSelectionConfig)
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