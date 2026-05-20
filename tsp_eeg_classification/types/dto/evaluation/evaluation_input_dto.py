from dataclasses import dataclass

from tsp_eeg_classification.types.dto.config.evaluation_config import EvaluationConfig
from tsp_eeg_classification.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from tsp_eeg_classification.types.dto.model.trained_model_dto import TrainedModelDTO
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO


@dataclass(frozen=True)
class EvaluationInputDTO:
    config: EvaluationConfig
    trained_models: list[TrainedModelDTO]
    folds: list[FoldDTO] # TODO: toto je i v tom DatasetSplitDTO
    dataset_split: DatasetSplitDTO | None = None
