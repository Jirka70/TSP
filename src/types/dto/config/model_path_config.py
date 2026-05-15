from pathlib import Path
from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class ModelPathConfig(AStageConfig):
    """Configuration for model paths."""

    backend: Literal["default"]
    path: Path
