from typing import Literal

from pydantic import Field, field_validator

from src.types.dto.config.astageconfig import AStageConfig


class VisualizationConfig(AStageConfig):
    """Configuration for the visualization stage."""

    backend: Literal["matplotlib", "plotly"]

    # Switches for different pipeline stages
    visualize_raw: bool
    visualize_raw_augmentation: bool
    visualize_epochs: bool
    visualize_augmentation: bool
    visualize_evaluation: bool

    # Global settings
    n_fft: int = Field(default=256, ge=1)
    save_plots: bool
    show_plots: bool

    @field_validator("n_fft")
    @classmethod
    def validate_n_fft_power_of_two(cls, v: int) -> int:
        """Ensures that n_fft is a power of two."""
        if (v & (v - 1)) != 0:
            raise ValueError(f"n_fft ({v}) must be a power of two (e.g., 256, 512, 1024).")
        return v
