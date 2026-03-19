from typing import Literal

from pydantic import Field, ImportString

from src.types.dto.config.astageconfig import AStageConfig


class PreprocessingConfigMNE(AStageConfig):
    backend: Literal["mne"]
    l_freq: float = Field(ge=0)
    h_freq: float = Field(ge=0)
    notch_freq: float | None
    sampling_rate_hz: float | None
    rereference: str | None
    channel_selection: list[str] | None
