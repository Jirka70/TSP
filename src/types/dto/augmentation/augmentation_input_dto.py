from dataclasses import dataclass

from src.types.dto.config.augmentation_config import AugmentationConfigBasic, AugmentationConfigTorchEEG
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO


@dataclass(frozen=True)
class AugmentationInputDTO:
    """
    Input data for the augmentation stage of the pipeline.

    Attributes:
        augmentation_config: The configuration for the augmentation backend (Basic or TorchEEG).
        data: The dataset splits (folds) to be augmented.
    """

    augmentation_config: AugmentationConfigBasic | AugmentationConfigTorchEEG
    data: DatasetSplitDTO
