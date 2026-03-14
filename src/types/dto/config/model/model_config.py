from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from src.types.dto.config.model.training_config import TrainingConfig


class ModelConfig(BaseModel):
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
