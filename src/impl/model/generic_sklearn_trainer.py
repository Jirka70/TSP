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
from src.types.interfaces.model.model_trainer import IModelTrainer

log = logging.getLogger(__name__)


class GenericSklearnTrainer(IModelTrainer):
    """
    A universal orchestrator for training Scikit-learn-based BCI classification pipelines.

    This trainer manages the lifecycle of a training run, from model instantiation
    via the factory to final metric collection. It is designed to work seamlessly
    with any pipeline that adheres to the Scikit-learn API, handling data extraction
    and basic validation performance tracking.
    """

    def run(self, input_dto: TrainingInputDTO, run_ctx: RunContext) -> StepResult[TrainedModelDTO]:
        """
        Executes the full training loop for a Scikit-learn model configuration.

        The method performs the following orchestration steps:
        1. Resolves model parameters from the configuration DTO.
        2. Instantiates the specific Pipeline and wraps it in a GenericSklearnModel.
        3. Extracts and validates training (and optional validation) data shapes.
        4. Triggers the fitting process.
        5. Captures training history and accuracy metrics for the final DTO.

        Args:
            input_dto (TrainingInputDTO): DTO containing model config, augmented
                training data, and optional validation sets.
            run_ctx (RunContext): Context metadata for the current execution.

        Returns:
            StepResult[TrainedModelDTO]: A standardized result containing the
                live model and its training performance summary.
        """
        method_id = input_dto.config.model_name
        params = getattr(input_dto.config, "parameters", getattr(input_dto.config, "metadata", {}))

        log.info(f"Start tréninku: {method_id} (Run: {run_ctx.run_id})")

        # Pipeline and model creation
        pipeline = ModelFactory.create(method_id, params)
        model = GenericSklearnModel(pipeline, method_id)

        # Data set-up
        x_train = self._extract_data(input_dto.train_data)
        y_train = np.asarray(input_dto.train_data.labels)

        # Fitting
        model.fit(x_train, y_train)

        # Metrics (train)
        train_acc = float(accuracy_score(y_train, model.predict(x_train)))

        # Validation (Optional)
        val_metrics = {}
        val_loss = []
        if input_dto.validation_data:
            x_val = self._extract_data(input_dto.validation_data)
            y_val = np.asarray(input_dto.validation_data.labels)
            val_acc = float(accuracy_score(y_val, model.predict(x_val)))
            val_metrics = {"accuracy": [val_acc]}
            val_loss = [1.0 - val_acc]

        # Result creation
        history = TrainingHistory(train_loss=[1.0 - train_acc], validation_loss=val_loss, train_metrics={"accuracy": [train_acc]}, validation_metrics=val_metrics)

        return StepResult(
            TrainedModelDTO(model=model, model_name=method_id, history=history, best_epoch=0, best_validation_metric_name="accuracy" if val_metrics else None, best_validation_metric_value=val_metrics["accuracy"][0] if val_metrics else None)
        )

    def _extract_data(self, data_dto: EpochPreprocessedDTO) -> np.ndarray:
        """
        Validates and extracts signal arrays from the preprocessing DTO.

        Ensures that the data follows the expected dimensionality for EEG classification:
        - 3D Arrays: $(N_{epochs}, N_{channels}, N_{times})$
        - 2D Arrays: $(N_{epochs}, N_{features})$ (for already extracted features)

        Args:
            data_dto (EpochPreprocessedDTO): The input data container.

        Returns:
            np.ndarray: The raw signal array ready for the model's fit/predict methods.

        Raises:
            ValueError: If the input array dimensions do not match the 2D or 3D requirements.
        """
        arr = data_dto.signal

        # Dimension check
        if arr.ndim == 3:
            # Standard EEG epochs (n_epochs, n_channels, n_times)
            return arr
        elif arr.ndim == 2:
            # Already extracted (n_epochs, n_features)
            log.info(f"Training on already extracted sings: {arr.shape}")
            return arr
        else:
            raise ValueError(f"2D (epochs, features) or 3D (epochs, channels, times) expected, but get: {arr.shape}")
