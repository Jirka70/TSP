# src/types/dto/epoching/epoching_input_dto.py

from dataclasses import dataclass

from src.types.dto.config.epoching_config import EpochingConfig
from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO


@dataclass(frozen=True)
class EpochingInputDTO:
    config: EpochingConfig
    preprocessed_data: PreprocessedDataDTO
