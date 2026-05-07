from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class RawAugmentationConfigNone(AStageConfig):
    """Configuration for disabling raw signal augmentation."""

    enabled: bool
    backend: Literal["none"]


class RawAugmentationConfigTorchEEG(AStageConfig):
    """Configuration for TorchEEG-based raw signal augmentation."""

    enabled: bool
    backend: Literal["torcheeg"]

    # Common TorchEEG parameters
    random_seed: int
    copies_per_sample: int

    # Transform specific parameters
    gaussian_noise_std: float
    mask_prob: float
    mask_ratio: float
    sign_flip_prob: float
    scale_prob: float
    scale_min: float
    scale_max: float
