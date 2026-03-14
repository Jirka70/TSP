from dataclasses import dataclass, field
from typing import Any

from src.types.dto.model.train_history import TrainingHistory


@dataclass(frozen=True)
class TrainedModelDTO:
    model: Any
    """
    Backend model object.
    E.g.. PyTorch model, sklearn model etc.
    """

    model_name: str
    """
    E.g. 'EEGNet'.
    """

    history: TrainingHistory | None = None

    best_epoch: int | None = None
    """
    Index of best epoch according to validation metrics
    """

    best_validation_metric_name: str | None = None
    best_validation_metric_value: float | None = None

    metadata: dict[str, object] = field(default_factory=dict)
