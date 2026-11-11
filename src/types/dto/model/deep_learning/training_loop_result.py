from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TrainingLoopResult:
    best_epoch: int | None
    best_validation_accuracy: float | None
    best_state_dict: dict[str, Tensor] | None
