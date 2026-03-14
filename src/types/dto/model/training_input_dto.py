from dataclasses import dataclass

from src.types.dto.config.model.model_config import ModelConfig
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO


@dataclass(frozen=True)
class TrainingInputDTO:

    config: ModelConfig
    train_data: EpochingDataDTO
    validation_data: EpochingDataDTO | None = None