import logging

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.train_history import TrainingHistory
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.interfaces.model.model import IModel
from src.types.interfaces.model.model_trainer import IModelTrainer

_MODEL_NAME = "riemannian_lda"

log = logging.getLogger(__name__)


class RiemannianModel(IModel):
    """
    Wrapper around a PyRiemann + scikit-learn Pipeline.

    The internal pipeline:
        Covariances (OAS) -> TangentSpace (Riemann) -> StandardScaler -> LDA
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline = Pipeline(
            steps=[
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("scaler", StandardScaler()),
                (
                    "lda",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        )

    def name(self) -> str:
        return _MODEL_NAME

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the pipeline.

        Parameters
        ----------
        x:
            EEG epochs with shape ``(n_epochs, n_channels, n_times)``.
        y:
            Integer class labels with shape ``(n_epochs,)``.
        """
        self._pipeline.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        return self._pipeline.predict(x)

    def predict_class_probability(self, x: np.ndarray) -> np.ndarray:
        """Return per-class probabilities via ``predict_proba``."""
        return self._pipeline.predict_proba(x)

    def get_state_dict(self) -> dict[str, Pipeline]:
        """Return a dict containing the fitted pipeline for serialisation."""
        return {"pipeline": self._pipeline}


def _extract_epochs_array(epoch_dto: EpochPreprocessedDTO) -> np.ndarray:
    return np.asarray(epoch_dto.signal)


class RiemannianModelTrainer(IModelTrainer):
    """
    Pipeline stage that trains a Riemannian geometry-based LDA classifier.

    Expected input
    --------------
    ``TrainingInputDTO.train_data.data`` must be raw EEG epochs with shape
    ``(n_epochs, n_channels, n_times)`` — either as an ``mne.Epochs`` object
    or a 3-D numpy array.
    """

    def run(
        self,
        input_dto: TrainingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[TrainedModelDTO]:
        log.info(
            "RiemannianModelTrainer: starting training "
            "(run_id=%s, source=%s)",
            run_ctx.run_id,
            run_ctx.dataset_name,
        )

        # ------------------------------------------------------------------ #
        # Prepare training data                                                #
        # ------------------------------------------------------------------ #
        x_train = _extract_epochs_array(input_dto.train_data)
        y_train = np.asarray(input_dto.train_data.labels)

        log.info(
            "Training data shape: %s, classes: %s",
            x_train.shape,
            np.unique(y_train).tolist(),
        )

        # ------------------------------------------------------------------ #
        # Fit                                                                  #
        # ------------------------------------------------------------------ #
        model = RiemannianModel()
        model.fit(x_train, y_train)

        # ------------------------------------------------------------------ #
        # Training metrics                                                     #
        # ------------------------------------------------------------------ #
        train_preds = model.predict(x_train)
        train_accuracy: float = float(accuracy_score(y_train, train_preds))
        log.info("Train accuracy: %.4f", train_accuracy)

        # ------------------------------------------------------------------ #
        # Validation metrics (optional)                                        #
        # ------------------------------------------------------------------ #
        val_loss_list: list[float] = []
        val_metrics: dict[str, list[float]] = {}

        if input_dto.validation_data is not None:
            x_val = _extract_epochs_array(input_dto.validation_data)
            y_val = np.asarray(input_dto.validation_data.labels)

            val_preds = model.predict(x_val)
            val_accuracy: float = float(accuracy_score(y_val, val_preds))

            # Use 1 - accuracy as a proxy loss so the history field is
            # populated consistently with iterative trainers.
            val_loss: float = 1.0 - val_accuracy
            val_loss_list = [val_loss]
            val_metrics = {"accuracy": [val_accuracy]}

            log.info(
                "Validation accuracy: %.4f  (proxy loss: %.4f)",
                val_accuracy,
                val_loss,
            )

        # ------------------------------------------------------------------ #
        # Build TrainingHistory (single-point — non-iterative model)          #
        # ------------------------------------------------------------------ #
        history = TrainingHistory(
            train_loss=[1.0 - train_accuracy],
            validation_loss=val_loss_list,
            train_metrics={"accuracy": [train_accuracy]},
            validation_metrics=val_metrics,
        )

        best_val_accuracy: float | None = (
            val_metrics["accuracy"][0] if val_metrics else None
        )

        trained_model_dto = TrainedModelDTO(
            model=model,
            model_name=_MODEL_NAME,
            history=history,
            best_epoch=0,
            best_validation_metric_name="accuracy" if best_val_accuracy is not None else None,
            best_validation_metric_value=best_val_accuracy,
        )

        log.info("RiemannianModelTrainer: training complete.")
        return StepResult(trained_model_dto)
