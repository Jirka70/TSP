from dataclasses import dataclass

from src.types.dto.config.model.model_config import ModelConfig, SklearnModelConfig
from src.types.dto.split.dataset_split_dto import FoldDTO


@dataclass(frozen=True)
class TrainingInputDTO:
    """Input data for the fold-based training stage."""

    config: ModelConfig | SklearnModelConfig
    folds: list[FoldDTO]
