from typing import Literal

from pydantic import BaseModel

from src.types.dto.config.astageconfig import AStageConfig


class AugmentationConfigBasic(AStageConfig):
    """Basic augmentation configuration for EEG samples (using numpy)"""

    backend: Literal["basic"]
    enabled: bool
    random_seed: int
    copies_per_sample: int
    gaussian_noise_std: float
    max_time_shift: int
    channel_dropout_prob: float


class AugmentationConfigTorchEEG(AStageConfig):
    """TorchEEG-based augmentation configuration for EEG samples."""

    backend: Literal["torcheeg"]
    enabled: bool
    random_seed: int
    copies_per_sample: int
    gaussian_noise_std: float
    mask_prob: float
    mask_ratio: float
    shift_prob: float
    sign_flip_prob: float
    scale_prob: float
    scale_min: float
    scale_max: float


class AugmentationConfigNone(BaseModel):
    """No augmentation configuration."""

    backend: Literal[None]
    enabled: bool
