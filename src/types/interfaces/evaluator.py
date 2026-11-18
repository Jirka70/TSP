from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.evaluation.evaluation_input_dto import (
    CrossValidationEvaluationInputDTO,
    ExperimentEvaluationInputDTO,
    TrainingEvaluationInputDTO,
)
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO


class ICrossValidationEvaluator(ABC):
    """
    Interface for fold-level cross-validation cross_validation (EvaluationPipeline).

    Implementations receive a list of trained models paired with the folds
    they were trained on, evaluate each model on its fold's test data, and
    return aggregated metrics across all folds.
    """

    @abstractmethod
    def run(
        self,
        input_dto: CrossValidationEvaluationInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EvaluationResultDTO]:
        raise NotImplementedError


class ITrainingEvaluator(ABC):
    """
    Interface for evaluating one final model on a held-out test set (TrainingPipeline).

    Implementations receive a single trained model and a dedicated test partition,
    compute predictions and metrics, and return the result.
    """

    @abstractmethod
    def run(
        self,
        input_dto: TrainingEvaluationInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EvaluationResultDTO]:
        raise NotImplementedError


class IExperimentEvaluator(ABC):
    """
    Interface for inference-only cross_validation (ExperimentPipeline).

    Implementations apply a pre-trained model to all supplied data.
    When labels are present in the data, metrics are computed; when they are
    absent, only predictions are returned (pure inference mode).
    """

    @abstractmethod
    def run(
        self,
        input_dto: ExperimentEvaluationInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EvaluationResultDTO]:
        raise NotImplementedError
