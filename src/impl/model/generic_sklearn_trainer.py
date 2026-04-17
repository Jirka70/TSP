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
    """
    A universal orchestrator for training Scikit-learn-based BCI classification pipelines.
    """

    def run(self, input_dto: TrainingInputDTO, run_ctx: RunContext) -> StepResult[TrainingResultDTO]:
        method_id = input_dto.config.model_name
        params = getattr(input_dto.config, "parameters", getattr(input_dto.config, "metadata", {}))

        log.info(f"Training started: {method_id} (Run: {run_ctx.run_id})")

        print(input_dto.folds.__len__())
        trained_models: list[TrainedModelDTO] = []
        for fold in input_dto.folds:
            # Pipeline and model creation
            pipeline = ModelFactory.create(method_id, params)
            model = GenericSklearnModel(pipeline, method_id)

            # Data set-up (extrahujeme X i y najednou ze všech nahrávek ve foldu)
            x_train, y_train = self._extract_data_and_labels(fold.train_data)

            # Fitting
            model.fit(x_train, y_train)

            # Metrics (train)
            train_acc = float(accuracy_score(y_train, model.predict(x_train)))

            # Validation (Optional)
            val_metrics = {}
            val_loss = []
            if fold.validation_data:
                x_val, y_val = self._extract_data_and_labels(fold.validation_data)
                val_acc = float(accuracy_score(y_val, model.predict(x_val)))
                val_metrics = {"accuracy": [val_acc]}
                val_loss = [1.0 - val_acc]

            # Result creation
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
            print(train_model.model_name, train_model.history.train_metrics["accuracy"], train_model.history.validation_metrics.get("accuracy", []))

        return StepResult(TrainingResultDTO(trained_models=trained_models))

    def _extract_data_and_labels(self, data_dto: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterates over all recordings in the DTO, extracts the signal arrays and labels from MNE Epochs, and concatenates them into a single training set for the fold.

        Args:
            data_dto (EpochPreprocessedDTO): The input data container with a list of recordings.

        Returns:
            tuple[np.ndarray, np.ndarray]: (X, y) where X is the feature array and y are the labels.

        Raises:
            ValueError: If the concatenated array dimensions do not match the 2D or 3D requirements.
        """
        x_list = []
        y_list = []

        # Iterujeme přes pole RecordingDTO
        for recording in data_dto.data:
            epochs = recording.data  # Zde očekáváme objekt mne.Epochs

            # Bezpečné získání dat z MNE objektu
            if hasattr(epochs, "get_data"):
                # MNE 1.0+ používá copy=False pro ušetření paměti
                x_list.append(epochs.get_data(copy=False))

                # Labely jsou ve třetím sloupci events matice
                y_list.append(epochs.events[:, -1])
            else:
                # Fallback, pokud by data uvnitř už byla numpy matice
                # V takovém případě musíš specifikovat, odkud brát labely (např. z metadata)
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))

        # Konkatenace listů do jedné velké matice podél osy epoch (axis=0)
        x_merged = np.concatenate(x_list, axis=0)
        y_merged = np.concatenate(y_list, axis=0)

        # Dimension check
        if x_merged.ndim == 3:
            # Standard EEG epochs (n_epochs, n_channels, n_times)
            pass
        elif x_merged.ndim == 2:
            # Already extracted (n_epochs, n_features)
            log.info(f"Training on already extracted signals: {x_merged.shape}")
        else:
            raise ValueError(f"2D (epochs, features) or 3D (epochs, channels, times) expected, but got: {x_merged.shape}")

        return x_merged, y_merged
