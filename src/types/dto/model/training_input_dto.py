from dataclasses import dataclass

from src.types.dto.config.model.model_config import ModelConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class TrainingInputDTO:
    config: ModelConfig
    train_data: EpochPreprocessedDTO
    validation_data: EpochPreprocessedDTO | None = None
