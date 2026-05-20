from dataclasses import dataclass

from tsp_eeg_classification.types.dto.config.split_config import SplitConfig
from tsp_eeg_classification.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class SplitInputDTO:
    config: SplitConfig
    data: EpochPreprocessedDTO
