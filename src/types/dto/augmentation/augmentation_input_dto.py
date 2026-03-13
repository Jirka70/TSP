from dataclasses import dataclass

from src.types.dto.config.augmentation_config import AugmentationConfig
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.training.prepared_samples_dto import PreparedSamplesDTO


@dataclass(frozen=True)
class AugmentationInputDTO:
    augmentationConfig: AugmentationConfig
    epoch_data: EpochingDataDTO
