from dataclasses import dataclass

from src.types.dto.config.split_config import SplitConfig
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO


@dataclass(frozen=True)
class SplitInputDTO:
    config: SplitConfig
    data: EpochingDataDTO
