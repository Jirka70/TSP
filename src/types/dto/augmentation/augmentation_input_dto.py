from dataclasses import dataclass

from src.types.dto.config.augmentation_config import AugmentationConfigBasic
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO


@dataclass(frozen=True)
class AugmentationInputDTO:
    augmentationConfig: AugmentationConfigBasic
    epoch_data: EpochingDataDTO
