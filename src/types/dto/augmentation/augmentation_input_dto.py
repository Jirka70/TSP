from dataclasses import dataclass

from src.types.dto.config.augmentation_config import AugmentationConfig
from src.types.dto.training.prepared_samples_dto import PreparedSamplesDTO


@dataclass(frozen=True)
class AugmentationInputDTO:
    samples: PreparedSamplesDTO
    augmentationConfig: AugmentationConfig
    enabled: bool = True
