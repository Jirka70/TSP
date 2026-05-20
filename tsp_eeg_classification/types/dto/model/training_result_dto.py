from dataclasses import dataclass, field

from tsp_eeg_classification.types.dto.model.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class TrainingResultDTO:
    """Result produced by the fold-based training stage."""

    trained_models: list[TrainedModelDTO] = field(default_factory=list)

