import copy
import logging

import numpy as np

from src.impl.model.deep_learning.eegnet_model import EEGNetModel
from src.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed
from src.impl.model.util.extract.extract_learning_data import extract_learning_data
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

log = logging.getLogger(__name__)


def extract_final_training_data(
        input_dto: FinalTrainingInputDTO,
) -> tuple[np.ndarray, np.ndarray]:
    if input_dto.training_data is not None:
        return extract_learning_data(input_dto.training_data)

    raise ValueError(
        "Final EEGNet training received no train_data. "
        "Falling back to fold-based training data reconstruction."
    )


def _evaluate_model(
        model: EEGNetModel,
        x_validation: np.ndarray | None,
        y_validation: np.ndarray | None,
) -> float | None:
    if x_validation is None or y_validation is None:
        log.warning("Validation data are not present. Model will not be validated")
        return None

    _, validation_accuracy = model.evaluate(x_validation, y_validation)

    return validation_accuracy


def _log_epoch_metrics(
        model: EEGNetModel,
        epoch: int,
) -> None:
    if model.history is None:
        log.warning("Epoch %s has no training history to summarize.", epoch + 1)
        return

    train_accuracy = model.history.train_metrics.get("accuracy", [])
    if train_accuracy:
        log.info(
            "Epoch %s train accuracy=%s",
            epoch + 1,
            train_accuracy[-1],
        )


def _train_one_epoch(
        model: EEGNetModel,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epoch: int,
) -> None:
    log.info("Epoch %s train", epoch + 1)
    model.train_one_epoch(x_train, y_train)


class FinalEEGNetTrainer(IFinalTrainer):
    """
    Trains one final EEGNet model.

    This trainer is not for fold-based evaluation. Prefer passing train_data that
    contains each final-training sample once. Cross-validation folds are accepted
    only as a backward-compatible fallback.
    """

    def run(
            self,
            input_dto: FinalTrainingInputDTO,
            run_ctx: RunContext,
    ) -> StepResult[FinalTrainingResultDTO]:
        if input_dto.training_data is None and not input_dto.folds:
            raise ValueError("Final EEGNet training needs training_data or at least one fold.")

        epochs = input_dto.config.training.epochs

        x_train, y_train = extract_final_training_data(input_dto)

        seed = input_dto.config.training.random_state
        if seed is not None:
            set_torch_seed(seed, input_dto.config.training.deterministic)

        network = create_eegnet_network(input_dto.config, x_train.shape)
        model = EEGNetModel(network=network,
                            model_name=input_dto.config.model_name,
                            config=input_dto.config)

        model.initialize_training(y_train)

        x_validation = None
        y_validation = None
        if input_dto.validation_data is not None:
            x_validation, y_validation = extract_learning_data(input_dto.validation_data)

        best_validation_accuracy: float | None = None
        best_epoch: int | None = None
        best_state: dict | None = None
        for epoch in range(epochs):
            log.info("Epoch %s/%s started", epoch + 1, epochs)
            _train_one_epoch(model, x_train, y_train, epoch)
            _log_epoch_metrics(model, epoch)
            log.info("Epoch %s evaluate validation data", epoch + 1)
            validation_accuracy = _evaluate_model(model, x_validation, y_validation)
            if validation_accuracy is not None:
                if best_validation_accuracy is None or validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.get_network_state_dict())
            log.info(
                "Epoch %s validation accuracy=%s",
                epoch + 1,
                validation_accuracy,
            )

        training_data_source = "train_data" if input_dto.training_data is not None else "fold_fallback"

        if best_state is not None:
            log.info("Applying the best state of model during training...")
            model.load_network_state_dict(best_state)
        else:
            best_epoch = epochs - 1

        trained_model = TrainedModelDTO(
            model=model,
            model_name=input_dto.config.model_name,
            history=model.history,
            best_epoch=best_epoch,
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
        return StepResult(
            FinalTrainingResultDTO(
                trained_model=trained_model,
            )
        )
