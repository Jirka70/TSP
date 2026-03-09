from dataclasses import dataclass

from src.types.dto.augmentation.augmented_samples import AugmentedSamplesDTO


@dataclass(frozen=True)
class TrainingInputDTO:
    samples: AugmentedSamplesDTO
    n_classes: int