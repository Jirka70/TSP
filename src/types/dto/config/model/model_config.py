from dataclasses import field
from typing import Any, Literal

from src.types.dto.config.astageconfig import AStageConfig
from src.types.dto.config.model.training_config import TrainingConfig


class ModelConfig(AStageConfig):
    _target_class = "impl.model.dummy_model_trainer.DummyModelTrainer"

    backend: Literal["eegnet"]

    n_classes: int
    n_channels: int
    n_times: int

    dropout: float
    kernel_length: int
    f1: int
    d: int
    f2: int

    training: TrainingConfig


class SklearnModelConfig(AStageConfig):
    backend: Literal["sklearn"]
    model_name: str

    parameters: dict[str, Any] = field(default_factory=dict)
