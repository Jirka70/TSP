from dataclasses import dataclass

from src.types.dto.config.split_config import SplitConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class SplitInputDTO:
    config: SplitConfig
    data: EpochPreprocessedDTO
