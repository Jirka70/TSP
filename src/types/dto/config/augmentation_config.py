from dataclasses import dataclass
from typing import Literal, Type

from pydantic import BaseModel, ConfigDict, ImportString

import src.impl.augmentation.dummy_augmentor


class AugmentationConfigBasic(BaseModel):
    backend: Literal["basic"]
    enabled: bool
    copies_per_sample: int
    gaussian_noise_std: float
    max_time_shift: int
    channel_dropout_prob: float

    stage: ImportString = "src.impl.augmentation.dummy_augmentor.DummyAugmentor"


class AugmentationConfigNone(BaseModel):
    backend: Literal[None]
    enabled: bool
