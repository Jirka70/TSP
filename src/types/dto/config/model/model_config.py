from dataclasses import dataclass

from src.types.dto.config.model.training_config import TrainingConfig


@dataclass(frozen=True)
class ModelConfig:
    backend: str

    n_classes: int
    n_channels: int
    n_times: int

    dropout: float
    kernel_length: int
    f1: int
    d: int
    f2: int

    training: TrainingConfig
