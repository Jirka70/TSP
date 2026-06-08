from pathlib import Path

from src.types.dto.config.astageconfig import AStageConfig


class ModelPathConfig(AStageConfig):
    """Configuration for model paths."""

    backend: str
    path: Path
