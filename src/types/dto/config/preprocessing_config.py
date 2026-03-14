from dataclasses import dataclass
from typing import Literal, Type

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from impl.preprocessing.dummy_preprocessing import DummyPreprocessing


class PreprocessingConfigMNE(BaseModel):
    backend: Literal["mne"]
    l_freq: float = Field(ge=0)
    h_freq: float = Field(ge=0)
    notch_freq: float | None
    sampling_rate_hz: float | None
    rereference: str | None
    channel_selection: list[str] | None

    # this line is needed for stage instance without validation
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # stage class which corresponds with this config
    stage: DummyPreprocessing = DummyPreprocessing()