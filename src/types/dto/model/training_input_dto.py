from dataclasses import dataclass

from src.types.dto.config.model.model_config import ModelConfig, SklearnModelConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import FoldDTO


@dataclass(frozen=True)
class TrainingInputDTO:
    """Input data for the fold-based training stage."""

    config: EEGNetConfig | SklearnModelConfig
    folds: list[FoldDTO]
    validation_data: EpochPreprocessedDTO | None = None
