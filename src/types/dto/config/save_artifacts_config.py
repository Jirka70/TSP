from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class SaveArtifactsConfig(AStageConfig):
    backend: Literal["default"]
    save_model: bool
    save_metrics: bool
    save_config: bool
    save_training_history: bool
