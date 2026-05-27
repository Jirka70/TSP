from dataclasses import dataclass
from typing import Any

from src.types.dto.config.model.model_config import EEGNetConfig


@dataclass(frozen=True)
class EEGNetModelState:
    model_name: str
    config: EEGNetConfig
    input_shape: tuple[int, int, int]
    network_state_dict: dict[str, Any]
    classes: list[Any] | None
    best_epoch: int | None
    best_validation_accuracy: float | None