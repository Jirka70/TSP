from dataclasses import dataclass

from src.types.dto.model.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class FinalTrainingResultDTO:
    """Result produced by the final training stage."""

    trained_model: TrainedModelDTO
