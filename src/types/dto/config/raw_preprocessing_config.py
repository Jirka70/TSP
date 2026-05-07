from typing import Literal, Optional
from pydantic import BaseModel
from src.types.dto.config.astageconfig import AStageConfig


class ResamplingConfig(BaseModel):
    enabled: bool = False
    sfreq: float


class HighPassFilterConfig(BaseModel):
    enabled: bool = True
    l_freq: float


class LowPassFilterConfig(BaseModel):
    enabled: bool = False
    h_freq: float


class NotchFilterConfig(BaseModel):
    enabled: bool = True
    freqs: list[float]


class BadChannelsInterpolationConfig(BaseModel):
    enabled: bool = True


class ICAConfig(BaseModel):
    enabled: bool = False
    n_components: float | int = 0.95
    method: Literal["infomax", "fastica", "picard"] = "infomax"


class ReReferencingConfig(BaseModel):
    enabled: bool = True
    method: Literal["CSD", "AVERAGE"] = "CSD"


class AnnotateBreakConfig(BaseModel):
    enabled: bool = True
    min_break_duration: float


class RawPreprocessingConfig(AStageConfig):
    backend: Literal["testing"]
    resampling: ResamplingConfig
    high_pass_filter: HighPassFilterConfig
    low_pass_filter: LowPassFilterConfig
    notch_filter: NotchFilterConfig
    bad_channels_interpolation: BadChannelsInterpolationConfig
    ica: ICAConfig
    re_referencing: ReReferencingConfig
    annotate_break: AnnotateBreakConfig