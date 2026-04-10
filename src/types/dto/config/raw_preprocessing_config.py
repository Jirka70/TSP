from typing import Literal

from pydantic import BaseModel

from src.types.dto.config.astageconfig import AStageConfig


class HighPassFilterConfig(BaseModel):
    l_freq: float


class NotchFilterConfig(BaseModel):
    freqs: list[float]


class AnnotateBreakConfig(BaseModel):
    min_break_duration: float


# TODO: Maybe named it like MNE (using mne, later it can use different library) + targer class
class RawPreprocessingConfig(AStageConfig):
    backend: Literal["testing"]
    high_pass_filter: HighPassFilterConfig
    notch_filter: NotchFilterConfig
    annotate_break: AnnotateBreakConfig
