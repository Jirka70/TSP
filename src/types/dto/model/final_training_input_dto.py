from dataclasses import dataclass

from src.types.dto.config.model.model_config import EEGNetConfig, SklearnModelConfig
from src.types.dto.split.dataset_split_dto import FoldDTO


@dataclass(frozen=True)
class FinalTrainingInputDTO:
    """Input data for the final training stage."""

    config: EEGNetConfig | SklearnModelConfig
    folds: list[FoldDTO]
