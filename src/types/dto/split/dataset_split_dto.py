from dataclasses import dataclass

from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class DatasetSplitDTO:
    train_data: EpochPreprocessedDTO
    validation_data: EpochPreprocessedDTO | None
    test_data: EpochPreprocessedDTO | None
