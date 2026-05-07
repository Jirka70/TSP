from dataclasses import dataclass

from src.types.dto.config.model.model_config import ModelConfig, EEGNetConfig, SklearnModelConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import FoldDTO


@dataclass(frozen=True)
class FinalTrainingInputDTO:
    """Input data for the final training stage."""

    config: EEGNetConfig | SklearnModelConfig
    folds: list[FoldDTO]
    train_data: EpochPreprocessedDTO | None = None
    validation_data: EpochPreprocessedDTO | None = None
