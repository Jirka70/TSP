import logging

import numpy as np
from sklearn.metrics import accuracy_score

from src.impl.model.generic_sklearn_model import GenericSklearnModel
from src.impl.model.model_factory import ModelFactory
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.train_history import TrainingHistory
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.interfaces.model.model_trainer import IModelTrainer

log = logging.getLogger(__name__)


class GenericSklearnTrainer(IModelTrainer):
    """Train Scikit-learn-based BCI pipelines across all configured folds."""

    def run(self, input_dto: TrainingInputDTO, run_ctx: RunContext) -> StepResult[TrainingResultDTO]:
        """Train the configured model for each fold and return the collected results."""
        if not input_dto.config.fold_training:
            log.info("Fold training is disabled. Skipping fold training stage.")
            return StepResult(TrainingResultDTO(trained_models=[]))

        method_id = input_dto.config.model_name
        params = getattr(input_dto.config, "parameters", getattr(input_dto.config, "metadata", {}))

        log.info(f"Training started: {method_id} (Run: {run_ctx.run_id})")

        log.info(f"Number of folds: {len(input_dto.folds)}")
        trained_models: list[TrainedModelDTO] = []
        for fold in input_dto.folds:
            pipeline = ModelFactory.create(method_id, params)
            model = GenericSklearnModel(pipeline, method_id)

            # Extract X and y from all recordings in the fold.
            x_train, y_train = self._extract_data_and_labels(fold.train_data)

            model.fit(x_train, y_train)

            train_acc = float(accuracy_score(y_train, model.predict(x_train)))

            val_metrics = {}
            val_loss = []
            if input_dto.validation_data:
                log.info("Validation data exists.")
                x_val, y_val = self._extract_data_and_labels(input_dto.validation_data)
                val_acc = float(accuracy_score(y_val, model.predict(x_val)))
                val_metrics = {"accuracy": [val_acc]}
                val_loss = [1.0 - val_acc]

            history = TrainingHistory(train_loss=[1.0 - train_acc], validation_loss=val_loss, train_metrics={"accuracy": [train_acc]}, validation_metrics=val_metrics)

            trained_models.append(
                TrainedModelDTO(
                    model=model,
                    model_name=method_id,
                    history=history,
                    best_epoch=0,
                    best_validation_metric_name="accuracy" if val_metrics else None,
                    best_validation_metric_value=val_metrics["accuracy"][0] if val_metrics else None,
                    fold_idx=fold.fold_idx,
                )
            )

        for train_model in trained_models:
            log.info(
                "Trained model %s: train accuracy=%s, validation accuracy=%s",
                train_model.model_name,
                train_model.history.train_metrics["accuracy"],
                train_model.history.validation_metrics.get("accuracy", []),
            )

        return StepResult(TrainingResultDTO(trained_models=trained_models))

    def _extract_data_and_labels(self, data_dto: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract signal arrays and labels from MNE Epochs and concatenate them into one training set.

        Args:
            data_dto (EpochPreprocessedDTO): Input data container with a list of recordings.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the feature array and labels.

        Raises:
            ValueError: If the concatenated feature array does not have a supported shape.
        """
        x_list = []
        y_list = []

        for recording in data_dto.data:
            epochs = recording.data

            if hasattr(epochs, "get_data"):
                x_list.append(epochs.get_data(copy=False))

                y_list.append(epochs.events[:, -1])
            else:
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))

        x_merged = np.concatenate(x_list, axis=0)
        y_merged = np.concatenate(y_list, axis=0)

        if x_merged.ndim == 3:
            pass
        elif x_merged.ndim == 2:
            log.info(f"Training on already extracted signals: {x_merged.shape}")
        else:
            raise ValueError(f"2D (epochs, features) or 3D (epochs, channels, times) expected, but got: {x_merged.shape}")

        return x_merged, y_merged
