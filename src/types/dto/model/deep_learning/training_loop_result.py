from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingLoopResult:
    best_epoch: int | None
    best_validation_accuracy: float | None
    best_state_dict: dict | None