from dataclasses import dataclass

from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO


@dataclass(frozen=True)
class EvaluationInputDTO:
    config: EvaluationConfig
    trained_models: list[TrainedModelDTO]
    folds: list[FoldDTO]
    dataset_split: DatasetSplitDTO | None = None
