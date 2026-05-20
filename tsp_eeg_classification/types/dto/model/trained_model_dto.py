from dataclasses import dataclass, field

from tsp_eeg_classification.types.dto.model.train_history import TrainingHistory
from tsp_eeg_classification.types.interfaces.model.model import IModel


@dataclass(frozen=True)
class TrainedModelDTO:
    """Trained model together with its training history and metadata."""

    model: IModel

    model_name: str

    history: TrainingHistory | None = None

    best_epoch: int | None = None

    best_validation_metric_name: str | None = None
    best_validation_metric_value: float | None = None
    fold_idx: int | None = None

    metadata: dict[str, object] = field(default_factory=dict)
