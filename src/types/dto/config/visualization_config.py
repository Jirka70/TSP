from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class VisualizationConfig(AStageConfig):
    """Configuration for the visualization stage."""

    backend: Literal["matplotlib", "plotly"]
    
    # Switches for different pipeline stages
    visualize_raw: bool = False
    visualize_epochs: bool = False
    visualize_augmentation: bool = False
    visualize_evaluation: bool = True
    
    # Global settings
    width: int = 12
    height: int = 6
    save_plots: bool = True
    show_plots: bool = False
