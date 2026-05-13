import logging

import numpy as np

from src.impl.model.deep_learning.eegnet_model import EEGNetModel
from src.impl.model.util.extract.extract_learning_data import extract_learning_data
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.split.dataset_split_dto import FoldDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

log = logging.getLogger(__name__)
NO_ACCURACY = 0

def extract_final_training_data_from_folds(
        folds: list[FoldDTO]
) -> tuple[np.ndarray, np.ndarray]:
    if not folds:
        raise ValueError("Final EEGNet training needs at least one fold.")

    unique_recordings = []
    # dataset_name, subject_id, session_id, run_id
    seen_recordings: set[tuple[object, object, object, object]] = set()

    for fold in folds:
        for recording in fold.train_data.data:
            key = (
                recording.dataset_name,
                recording.subject_id,
                recording.session_id,
                recording.run_id
            )

            if key in seen_recordings:
                continue

            seen_recordings.add(key)
            unique_recordings.append(recording)

    if not unique_recordings:
        raise ValueError("No unique training data found in folds.")

    return extract_learning_data(EpochPreprocessedDTO(data=unique_recordings))


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
        epochs: int = input_dto.config.training.epochs
        x_train, y_train = extract_final_training_data_from_folds(input_dto.folds)

        network = create_eegnet_network(input_dto.config, x_train.shape)
        model = EEGNetModel(network=network,
                            model_name=input_dto.config.model_name,
                            config=input_dto.config)

        model.initialize_training(y_train)
        for _ in range(epochs):
            self._train_one_epoch(model, x_train, y_train, run_ctx)
            validation_accuracy = self._evaluate_model(model, input_dto.validation_data)

        trained_model = TrainedModelDTO(
            model=model,
            model_name=input_dto.config.model_name,
            history=model.history,
            best_epoch=None,
            best_validation_metric_name=None,
            best_validation_metric_value=None,
            fold_idx=None,
            metadata={
                "training_mode": "final_fold_training",
                "run_id": run_ctx.run_id,
                "n_folds": len(input_dto.folds),
            },
        )
        return StepResult(
            FinalTrainingResultDTO(
                trained_model=trained_model,
            )
        )

    def _train_one_epoch(self, model: EEGNetModel, x_train: np.ndarray, y_train: np.ndarray, run_ctx: RunContext):
        model.train_one_epoch(x_train, y_train)

    def _evaluate_model(self, model: EEGNetModel, validation_data: EpochPreprocessedDTO | None) -> float:
        if validation_data is None:
            log.warning("Validation data are not present. Model will not be validated")
            return NO_ACCURACY

        x_validation, y_validation = extract_learning_data(validation_data)
        _, validation_accuracy = model.evaluate(x_validation, y_validation)

        return validation_accuracy
