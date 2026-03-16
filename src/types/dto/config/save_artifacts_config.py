from dataclasses import dataclass

from src.types.dto.config.astageconfig import AStageConfig


@dataclass(frozen=True)
class SaveArtifactsConfig(AStageConfig):
    _target_class = "impl.artifacts_saver.artifacts_saver.ArtifactSaver"
    save_model: bool
    save_metrics: bool
    save_config: bool
    save_training_history: bool
