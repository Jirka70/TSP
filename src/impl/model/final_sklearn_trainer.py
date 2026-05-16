import logging
from typing import List, Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.impl.model.generic_sklearn_model import GenericSklearnModel
from src.impl.model.model_factory import ModelFactory
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.train_history import TrainingHistory
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

log = logging.getLogger(__name__)


class FinalSklearnTrainer(IFinalTrainer):
    """Train a Scikit-learn model on all available folds for final export."""

    def run(self, input_dto: FinalTrainingInputDTO, run_ctx: RunContext) -> StepResult[FinalTrainingResultDTO]:
        """
        Fit the configured model on all available training data and wrap the result.

        Args:
            input_dto (FinalTrainingInputDTO): Input data container containing folds and configuration.
            run_ctx (RunContext): Context information for the current execution run.

        Returns:
            StepResult[FinalTrainingResultDTO]: Encapsulated final trained model artifact.
        """
        method_id = input_dto.config.model_name
        params = getattr(input_dto.config, "parameters", getattr(input_dto.config, "metadata", {}))

        log.info(f"Final training: {method_id} (Run: {run_ctx.run_id})")

        # Accumulate all training samples from the provided folds
        x_all, y_all = self._collect_all_data(input_dto)

        pipeline : Pipeline = ModelFactory.create(method_id, params)
        model : GenericSklearnModel = GenericSklearnModel(pipeline, method_id)
        model.fit(x_all, y_all)

        train_acc : float = float(accuracy_score(y_all, model.predict(x_all)))
        history = TrainingHistory(
            train_loss=[1.0 - train_acc],
            train_metrics={"accuracy": [train_acc]},
        )

        trained_model : TrainedModelDTO = TrainedModelDTO(
            model=model,
            model_name=method_id,
            history=history,
            best_epoch=0,
            best_validation_metric_name=None,
            best_validation_metric_value=None,
            fold_idx=None,
        )

        log.info(f"Final model trained: accuracy={train_acc:.4f} on {len(y_all)} samples")

        return StepResult(FinalTrainingResultDTO(trained_model=trained_model))

    def _collect_all_data(self, input_dto: FinalTrainingInputDTO) -> tuple[np.ndarray, np.ndarray]:
        """
        Collect and concatenate training samples and labels across all available folds.

        Args:
            input_dto (FinalTrainingInputDTO): Input container with fold data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Concatenated feature matrix (X) and labels (y).
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for fold in input_dto.folds:
            if fold.train_data:
                x, y = self._extract_data_and_labels(fold.test_data) # TODO: Chceme skutecne vyuzivat test_date nebo chceme spise train_data.
                x_list.append(x)
                y_list.append(y)
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

    def _extract_data_and_labels(self, data_dto: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from preprocessed epoch recordings.

        Args:
            data_dto (EpochPreprocessedDTO): Preprocessed data container.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and labels arrays.
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for recording in data_dto.data:
            epochs : Any = recording.data

            if hasattr(epochs, "get_data"):
                x_list.append(epochs.get_data(copy=False))
                y_list.append(epochs.events[:, -1])
            else:
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))

        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)
