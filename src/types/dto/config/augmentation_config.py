from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from impl.augmentation.dummy_augmentor import DummyAugmentor


class AugmentationConfigBasic(BaseModel):
    backend: Literal["basic"]
    enabled: bool
    copies_per_sample: int
    gaussian_noise_std: float
    max_time_shift: int
    channel_dropout_prob: float


class AugmentationConfigNone(BaseModel):
    backend: Literal[None]
    enabled: bool
