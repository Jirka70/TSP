from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class FinalTrainerConfig(AStageConfig):
    """Configuration for the final model training stage."""

    backend: Literal["sklearn"]
