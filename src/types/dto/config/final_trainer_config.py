from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class FinalTrainerConfig(AStageConfig):
    backend: Literal["sklearn"]
