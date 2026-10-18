from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.types.dto.config.astageconfig import AStageConfig


class GaussianNoiseConfig(BaseModel):
    std: float = Field(ge=0.0, description="Standard deviation of Gaussian noise.")


class MaskConfig(BaseModel):
    prob: float = Field(ge=0.0, le=1.0, description="Probability of applying a random mask.")
    ratio: float = Field(ge=0.0, le=1.0, description="Ratio of the signal to be masked.")


class SignFlipConfig(BaseModel):
    prob: float = Field(ge=0.0, le=1.0, description="Probability of inverting the sign.")


class ScaleConfig(BaseModel):
    prob: float = Field(ge=0.0, le=1.0, description="Probability of scaling the amplitude.")
    min: float = Field(gt=0.0, description="Minimum factor for amplitude scaling.")
    max: float = Field(gt=0.0, description="Maximum factor for amplitude scaling.")


class RawAugmentationConfigNone(AStageConfig):
    """Configuration for disabling raw signal augmentation."""

    enabled: bool = Field(description="Whether to enable raw augmentation.")
    backend: Literal["none"] = Field(description="Backend identifier.")


class RawAugmentationConfigTorchEEG(AStageConfig):
    """Configuration for TorchEEG-based raw signal augmentation."""

    enabled: bool = Field(description="Whether to enable raw augmentation.")
    backend: Literal["raw_torcheeg"] = Field(description="Backend identifier.")

    random_seed: int = Field(ge=0, description="Random seed for reproducibility.")
    copies_per_sample: int = Field(ge=0, description="Number of augmented copies to create.")

    gaussian_noise: GaussianNoiseConfig = Field(description="Gaussian noise transform settings.")
    mask: MaskConfig = Field(description="Random mask transform settings.")
    sign_flip: SignFlipConfig = Field(description="Sign flip transform settings.")
    scale: ScaleConfig = Field(description="Amplitude scale transform settings.")

    @model_validator(mode="after")
    def validate_scale_range(self) -> "RawAugmentationConfigTorchEEG":
        if self.scale.min > self.scale.max:
            raise ValueError(f"scale.min ({self.scale.min}) must be less than or equal to scale.max ({self.scale.max})")
        return self
