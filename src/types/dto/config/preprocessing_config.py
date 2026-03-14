from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class PreprocessingConfigMNE(BaseModel):
    backend: Literal["mne"]
    l_freq: float | None
    h_freq: float = Field(ge=0)
    notch_freq: float | None
    sampling_rate_hz: float | None
    rereference: str | None
    channel_selection: list[str] | None