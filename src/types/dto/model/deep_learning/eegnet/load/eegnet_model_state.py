from dataclasses import dataclass

from torch import Tensor

from src.types.dto.config.model.model_config import EEGNetConfig


@dataclass(frozen=True)
class EEGNetModelState:
    model_name: str
    config: EEGNetConfig
    input_shape: tuple[int, int, int]
    network_state_dict: dict[str, Tensor]
    classes: list[object] | None
    best_epoch: int | None
    best_validation_accuracy: float | None
