from dataclasses import dataclass

from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class EvaluationInputDTO:
    config: EvaluationConfig
    trained_model: TrainedModelDTO
    test_data: EpochingDataDTO