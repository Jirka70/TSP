from dataclasses import dataclass

from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class DLInputDTO:
    """Internal EEGNet input with explicit train, validation and test sets."""

    config: EEGNetConfig
    train: EpochPreprocessedDTO
    val: EpochPreprocessedDTO
    test: EpochPreprocessedDTO
