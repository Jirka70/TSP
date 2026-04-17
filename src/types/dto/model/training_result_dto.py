from dataclasses import dataclass, field

from src.types.dto.model.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class TrainingResultDTO:
    trained_models: list[TrainedModelDTO] = field(default_factory=list)

