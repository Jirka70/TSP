from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.types.dto.config.astageconfig import AStageConfig


class AugmentationConfigBasic(AStageConfig):
    """Basic augmentation configuration for EEG samples (using numpy)"""

    backend: Literal["basic"]
    enabled: bool
    random_seed: int = Field(default=42)
    copies_per_sample: int = Field(default=1, ge=1)
    gaussian_noise_std: float = Field(default=0.0, ge=0.0)
    max_time_shift: int = Field(default=0, ge=0)
    channel_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class AugmentationConfigTorchEEG(AStageConfig):
    """TorchEEG-based augmentation configuration for EEG samples."""

    backend: Literal["torcheeg"]
    enabled: bool
    random_seed: int = Field(default=42)
    copies_per_sample: int = Field(default=1, ge=1)
    gaussian_noise_std: float = Field(default=0.0, ge=0.0)
    mask_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    mask_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    shift_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    sign_flip_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    scale_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    scale_min: float = Field(default=1.0)
    scale_max: float = Field(default=1.0)

    @model_validator(mode="after")
    def validate_scale_range(self) -> "AugmentationConfigTorchEEG":
        """Ensures that scale_min is not greater than scale_max."""
        if self.scale_min > self.scale_max:
            raise ValueError(f"scale_min ({self.scale_min}) must be less than or equal to scale_max ({self.scale_max})")
        return self


class AugmentationConfigNone(BaseModel):
    """No augmentation configuration."""

    backend: Literal[None]
    enabled: bool
