from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class VisualizationConfig(AStageConfig):
    """Configuration for the visualization stage."""

    backend: Literal["matplotlib", "plotly"]

    # Switches for different pipeline stages
    visualize_raw: bool
    visualize_epochs: bool
    visualize_augmentation: bool
    visualize_evaluation: bool

    # Global settings
    width: int
    height: int
    n_fft: int = 256
    save_plots: bool
    show_plots: bool = False
