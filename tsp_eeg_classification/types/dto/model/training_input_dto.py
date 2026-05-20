from dataclasses import dataclass

from tsp_eeg_classification.types.dto.config.model.model_config import EEGNetConfig, SklearnModelConfig
from tsp_eeg_classification.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from tsp_eeg_classification.types.dto.split.dataset_split_dto import FoldDTO


@dataclass(frozen=True)
class TrainingInputDTO:
    """Input data for the fold-based training stage."""

    config: EEGNetConfig | SklearnModelConfig
    folds: list[FoldDTO]
