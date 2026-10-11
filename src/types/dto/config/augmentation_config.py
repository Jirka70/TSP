from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.types.dto.config.astageconfig import AStageConfig


class GaussianNoiseConfig(BaseModel):
    std: float = Field(default=0.0, ge=0.0)


class MaskConfig(BaseModel):
    prob: float = Field(default=0.0, ge=0.0, le=1.0)
    ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ShiftConfig(BaseModel):
    prob: float = Field(default=0.0, ge=0.0, le=1.0)


class SignFlipConfig(BaseModel):
    prob: float = Field(default=0.0, ge=0.0, le=1.0)


class ScaleConfig(BaseModel):
    prob: float = Field(default=0.0, ge=0.0, le=1.0)
    min: float = Field(default=1.0)
    max: float = Field(default=1.0)


class TimeShiftConfig(BaseModel):
    max_samples: int = Field(default=0, ge=0)


class ChannelDropoutConfig(BaseModel):
    prob: float = Field(default=0.0, ge=0.0, le=1.0)


class AugmentationConfigBasic(AStageConfig):
    """Basic augmentation configuration for EEG samples (using numpy)"""

    backend: Literal["basic"]
    enabled: bool
    random_seed: int = Field(default=42)
    copies_per_sample: int = Field(default=1, ge=1)
    gaussian_noise: GaussianNoiseConfig = Field(default_factory=GaussianNoiseConfig)
    time_shift: TimeShiftConfig = Field(default_factory=TimeShiftConfig)
    channel_dropout: ChannelDropoutConfig = Field(default_factory=ChannelDropoutConfig)


class AugmentationConfigTorchEEG(AStageConfig):
    """TorchEEG-based augmentation configuration for EEG samples."""

    backend: Literal["torcheeg"]
    enabled: bool
    random_seed: int = Field(default=42)
    copies_per_sample: int = Field(default=1, ge=1)
    gaussian_noise: GaussianNoiseConfig = Field(default_factory=GaussianNoiseConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    shift: ShiftConfig = Field(default_factory=ShiftConfig)
    sign_flip: SignFlipConfig = Field(default_factory=SignFlipConfig)
    scale: ScaleConfig = Field(default_factory=ScaleConfig)

    @model_validator(mode="after")
    def validate_scale_range(self) -> "AugmentationConfigTorchEEG":
        """Ensures that scale.min is not greater than scale.max."""
        if self.scale.min > self.scale.max:
            raise ValueError(f"scale.min ({self.scale.min}) must be less than or equal to scale.max ({self.scale.max})")
        return self


class AugmentationConfigNone(BaseModel):
    """No augmentation configuration."""

    backend: Literal[None]
    enabled: bool
