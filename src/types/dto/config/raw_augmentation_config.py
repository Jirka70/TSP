from typing import Literal

from pydantic import Field, model_validator

from src.types.dto.config.astageconfig import AStageConfig


class RawAugmentationConfigNone(AStageConfig):
    """Configuration for disabling raw signal augmentation."""

    enabled: bool = Field(description="Whether to enable raw augmentation.")
    backend: Literal["none"] = Field(description="Backend identifier.")


class RawAugmentationConfigTorchEEG(AStageConfig):
    """Configuration for TorchEEG-based raw signal augmentation."""

    enabled: bool = Field(description="Whether to enable raw augmentation.")
    backend: Literal["raw_torcheeg"] = Field(description="Backend identifier.")

    # Common TorchEEG parameters
    random_seed: int = Field(ge=0, description="Random seed for reproducibility.")
    copies_per_sample: int = Field(ge=0, description="Number of augmented copies to create.")

    # Transform specific parameters
    gaussian_noise_std: float = Field(ge=0.0, description="Standard deviation of Gaussian noise.")
    mask_prob: float = Field(ge=0.0, le=1.0, description="Probability of applying a random mask.")
    mask_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of the signal to be masked.")
    sign_flip_prob: float = Field(ge=0.0, le=1.0, description="Probability of inverting the sign.")
    scale_prob: float = Field(ge=0.0, le=1.0, description="Probability of scaling the amplitude.")
    scale_min: float = Field(gt=0.0, description="Minimum factor for amplitude scaling.")
    scale_max: float = Field(gt=0.0, description="Maximum factor for amplitude scaling.")

    @model_validator(mode="after")
    def validate_scale_range(self) -> "RawAugmentationConfigTorchEEG":
        if self.scale_min > self.scale_max:
            raise ValueError(f"scale_min ({self.scale_min}) must be less than or equal to scale_max ({self.scale_max})")
        return self
