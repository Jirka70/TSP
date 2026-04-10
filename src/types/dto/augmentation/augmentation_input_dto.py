from dataclasses import dataclass

from src.types.dto.config.augmentation_config import AugmentationConfigBasic, AugmentationConfigTorchEEG
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class AugmentationInputDTO:
    augmentationConfig: AugmentationConfigBasic | AugmentationConfigTorchEEG
    epoch_data: EpochPreprocessedDTO
