from typing import Literal

from src.types.dto.config.model.training_config import TrainingConfig
from src.types.dto.config.astageconfig import AStageConfig


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
