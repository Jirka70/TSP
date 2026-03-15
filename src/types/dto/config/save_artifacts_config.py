# src/types/dto/config/artifacts_config.py

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class SaveArtifactsConfig(BaseModel):
    save_model: bool
    save_metrics: bool
    save_config: bool
    save_training_history: bool
