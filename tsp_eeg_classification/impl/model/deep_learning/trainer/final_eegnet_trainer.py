import copy

import numpy as np

from tsp_eeg_classification.impl.model.deep_learning.factory.deep_learning_model_factory import DeepLearningModelFactory
from tsp_eeg_classification.impl.model.deep_learning.eegnet_model import EEGNetModel
from tsp_eeg_classification.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed
from tsp_eeg_classification.impl.model.util.extract.extract_learning_data import extract_learning_data
from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.pipeline_logging.pipeline_logger import PipelineLogger
from tsp_eeg_classification.types.dto.config.logging.verbosity_level import VerbosityLevel
from tsp_eeg_classification.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from tsp_eeg_classification.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from tsp_eeg_classification.types.dto.model.trained_model_dto import TrainedModelDTO
from tsp_eeg_classification.types.interfaces.model.final_trainer import IFinalTrainer


def extract_final_training_data(
        input_dto: FinalTrainingInputDTO,
) -> tuple[np.ndarray, np.ndarray]:
    if input_dto.training_data is not None:
        return extract_learning_data(input_dto.training_data)

    raise ValueError(
        "Final EEGNet training received no train_data. "
        "Final EEGNet training requires training_data."
    )


def _evaluate_model(
        model: EEGNetModel,
        x_validation: np.ndarray | None,
        y_validation: np.ndarray | None,
        log: PipelineLogger
) -> float | None:
    if x_validation is None or y_validation is None:
        return None

    log.info("Evaluating validation data", VerbosityLevel.TRACE)
    _, validation_accuracy = model.evaluate(x_validation, y_validation)

    return validation_accuracy


def _log_epoch_metrics(
        model: EEGNetModel,
        log: PipelineLogger,
        epoch: int,
) -> None:
    if model.history is None:
        log.warning(f"Epoch {epoch + 1} has no training history to summarize.")
        return

    train_accuracy = model.history.train_metrics.get("accuracy", [])
    if train_accuracy:
        log.info(f"Epoch {epoch + 1} train accuracy={train_accuracy[-1]}", VerbosityLevel.DETAILED)


def _train_one_epoch(
        model: EEGNetModel,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epoch: int,
        log: PipelineLogger
) -> None:
    log.info(f"Epoch {epoch + 1} train", VerbosityLevel.DETAILED)
    model.train_one_epoch(x_train, y_train)


class FinalEEGNetTrainer(IFinalTrainer):
    """
    Trains one final EEGNet model.

    This trainer is not for fold-based evaluation. Prefer passing train_data that
    contains each final-training sample once. Cross-validation folds are accepted
    only as a backward-compatible fallback.
    """

    def __init__(self, model_factory: DeepLearningModelFactory | None = None) -> None:
        self._model_factory = model_factory or DeepLearningModelFactory()

    def run(
            self,
            input_dto: FinalTrainingInputDTO,
            run_ctx: RunContext,
    ) -> StepResult[FinalTrainingResultDTO]:
        log = run_ctx.logger.for_step("EEGNET FINAL TRAINING")
        log.info("Started training EEGNET model", VerbosityLevel.QUIET)
        epochs = input_dto.config.training.epochs

        log.info("Extracting training data", VerbosityLevel.DETAILED)
        x_train, y_train = extract_final_training_data(input_dto)
        log.info(f"Training data extracted: samples={len(y_train)} shape={x_train.shape}", VerbosityLevel.TRACE)

        seed = input_dto.config.training.random_state
        if seed is not None:
            set_torch_seed(seed, input_dto.config.training.deterministic)
            log.info("Seed from input config was applied successfully", VerbosityLevel.TRACE)
        else:
            log.info("Seed was set to None. No initial seed is being applied...", VerbosityLevel.TRACE)

        model = self._model_factory.create(config=input_dto.config, input_shape=x_train.shape)
        log.info("EEGNET model created successfully")

        model.initialize_training(y_train)
        log.info("Initializing EEGNET training...")

        x_validation = None
        y_validation = None
        if input_dto.validation_data is not None:
            log.info("Extracting validation data", VerbosityLevel.TRACE)
            x_validation, y_validation = extract_learning_data(input_dto.validation_data)
            log.info(f"Validation data extracted: samples={len(y_validation)} shape={x_validation.shape}", VerbosityLevel.TRACE)
        else:
            log.warning("Validation data are not present. Model will not be validated")

        best_validation_accuracy: float | None = None
        best_epoch: int | None = None
        best_state: dict | None = None
        for epoch in range(epochs):
            log.info(f"Training EEGNET model for epoch {epoch + 1}", VerbosityLevel.TRACE)
            _train_one_epoch(model, x_train, y_train, epoch, log)
            _log_epoch_metrics(model, log, epoch)
            validation_accuracy = _evaluate_model(model, x_validation, y_validation, log)
            if validation_accuracy is not None:
                if best_validation_accuracy is None or validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.get_network_state_dict())
            log.info(f"Epoch {epoch + 1} validation accuracy={validation_accuracy}", VerbosityLevel.DETAILED)

        training_data_source = "train_data" if input_dto.training_data is not None else "fold_fallback"

        if best_state is not None:
            log.info(f"Applying best model state from epoch {best_epoch + 1}.", VerbosityLevel.DETAILED)
            model.load_network_state_dict(best_state)
        else:
            best_epoch = epochs - 1

        trained_model = TrainedModelDTO(
            model=model,
            model_name=input_dto.config.model_name,
            history=model.history,
            best_epoch=best_epoch,
            fold_idx=None,
            best_validation_metric_name="accuracy" if best_validation_accuracy is not None else None,
            best_validation_metric_value=best_validation_accuracy,
            metadata={
                "training_mode": "final_training",
                "run_id": run_ctx.run_id,
                "n_folds": len(input_dto.folds),
                "n_train_samples": len(y_train),
                "n_validation_samples": len(y_validation) if y_validation is not None else 0,
                "training_data_source": training_data_source
            },
        )
        log.info("Finished training EEGNET model", VerbosityLevel.QUIET)
        return StepResult(
            FinalTrainingResultDTO(
                trained_model=trained_model,
            )
        )
