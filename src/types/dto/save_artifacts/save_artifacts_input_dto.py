# src/types/dto/artifacts/save_artifacts_input_dto.py

from dataclasses import dataclass

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.save_artifacts_config import SaveArtifactsConfig
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO


@dataclass(frozen=True)
class SaveArtifactsInputDTO:
    config: SaveArtifactsConfig
    experiment_config: ExperimentConfig
    """
    For saving whole pipeline configuration
    """
    run_context: RunContext
    trained_model: TrainedModelDTO | None = None
    evaluation_result: EvaluationResultDTO | None = None
