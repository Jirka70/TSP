from dataclasses import dataclass

from src.types.dto.config.raw_augmentation_config import (
    RawAugmentationConfigNone,
    RawAugmentationConfigTorchEEG,
)
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO


@dataclass(frozen=True)
class RawAugmentationInputDTO:
    """
    Data transfer object for the raw signal augmentation stage.
    """
    config: RawAugmentationConfigNone | RawAugmentationConfigTorchEEG
    data: RawPreprocessedDTO
