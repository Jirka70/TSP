from dataclasses import dataclass

from src.types.dto.config.evaluation_config import TrainingEvaluationConfig, CrossValidationEvaluationConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.split.dataset_split_dto import FoldDTO


@dataclass(frozen=True)
class CrossValidationEvaluationInputDTO:
    """
    Input DTO for fold-level cross-validation cross_validation (EvaluationPipeline).

    Carries a list of trained models (one per fold) and the corresponding folds.
    The evaluator pairs them by ``fold_idx``: for each fold it evaluates the model
    whose ``fold_idx`` matches ``fold.fold_idx`` on ``fold.test_data``.

    Attributes:
        config: Evaluation configuration (metrics to compute, etc.).
        trained_models: Models produced by the fold trainer, each with a ``fold_idx``.
        folds: Cross-validation folds produced by the splitter.
            Every fold must have ``test_data`` populated.
    """

    config: CrossValidationEvaluationConfig
    trained_models: list[TrainedModelDTO]
    folds: list[FoldDTO]


@dataclass(frozen=True)
class TrainingEvaluationInputDTO:
    """
    Input DTO for evaluating one final model on a held-out test set (TrainingPipeline).

    Attributes:
        config: Evaluation configuration (metrics to compute, etc.).
        trained_model: The single final model to evaluate.
        test_data: Held-out test partition that the model has never seen during training.
    """

    config: TrainingEvaluationConfig
    trained_model: TrainedModelDTO
    test_data: EpochPreprocessedDTO


@dataclass(frozen=True)
class ExperimentEvaluationInputDTO:
    """
    Input DTO for inference-only cross_validation (ExperimentPipeline).

    The model is applied to all supplied data.  Labels are optional:
    when present the evaluator computes metrics; when absent it only
    returns predictions (pure inference mode).

    Attributes:
        config: Evaluation configuration (metrics to compute, etc.).
        trained_model: Pre-trained model loaded from disk.
        data: All available epoched data to run inference on.
    """

    config: TrainingEvaluationConfig
    trained_model: TrainedModelDTO
    data: EpochPreprocessedDTO
