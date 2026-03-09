from dataclasses import dataclass

from src.types.dto.augmentation.augmented_samples_dto import AugmentedSamplesDTO
from src.types.dto.training.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class EvaluationInputDTO:
    trained_model: TrainedModelDTO
    samples: AugmentedSamplesDTO
    metrics: list[str]