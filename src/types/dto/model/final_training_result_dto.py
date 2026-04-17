from dataclasses import dataclass

from src.types.dto.model.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class FinalTrainingResultDTO:
    trained_model: TrainedModelDTO
