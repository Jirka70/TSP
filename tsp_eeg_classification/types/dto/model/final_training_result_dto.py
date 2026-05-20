from dataclasses import dataclass

from tsp_eeg_classification.types.dto.model.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class FinalTrainingResultDTO:
    """Result produced by the final training stage."""

    trained_model: TrainedModelDTO
