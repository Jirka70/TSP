import importlib
from typing import Literal

from pydantic import BaseModel

from src.types.dto.config.astageconfig import AStageConfig


class AugmentationConfigBasic(AStageConfig):
    _target_class = "src.impl.augmentation.dummy_augmentor.DummyAugmentor"

    backend: Literal["basic"]
    enabled: bool
    copies_per_sample: int
    gaussian_noise_std: float
    max_time_shift: int
    channel_dropout_prob: float


class AugmentationConfigNone(BaseModel):
    _target_class = "src.impl.augmentation.dummy_augmentor.DummyAugmentor"

    backend: Literal[None]
    enabled: bool