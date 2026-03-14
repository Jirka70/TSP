from dataclasses import dataclass

from src.types.dto.config.augmentation_config import AugmentationConfig
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO


@dataclass(frozen=True)
class AugmentationInputDTO:
    augmentationConfig: AugmentationConfig
    epoch_data: EpochingDataDTO
