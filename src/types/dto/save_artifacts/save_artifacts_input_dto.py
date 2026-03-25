from dataclasses import dataclass
from pathlib import Path

from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.save_artifacts_config import SaveArtifactsConfig
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.model_serializer import IModelSerializer


@dataclass(frozen=True)
class SaveArtifactsInputDTO:
    config: SaveArtifactsConfig
    experiment_config: ExperimentConfig
    """
    For saving whole pipeline configuration
    """
    output_path: Path
    model_serializer: IModelSerializer
    trained_model: TrainedModelDTO | None = None
    evaluation_result: EvaluationResultDTO | None = None
