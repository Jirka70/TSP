from dataclasses import dataclass

from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO


@dataclass(frozen=True)
class DatasetSplitDTO:

    train_data: EpochingDataDTO
    validation_data: EpochingDataDTO | None
    test_data: EpochingDataDTO | None

