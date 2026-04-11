import numpy as np
from sklearn.pipeline import Pipeline

from src.types.interfaces.model.model import IModel


class GenericSklearnModel(IModel):
    """
    A concrete implementation of the IModel interface wrapping a Scikit-learn Pipeline.

    This class serves as a standardized wrapper for any Scikit-learn compatible
    pipeline, providing a unified API for training, inference, and serialization
    within the EEG processing framework.
    """

    def __init__(self, pipeline: Pipeline, model_name: str) -> None:
        """
        Initializes the model with a specific pipeline and identifier.

        Args:
            pipeline (Pipeline): The Scikit-learn Pipeline instance to wrap.
            model_name (str): The unique name or identifier for this model configuration.
        """
        self._pipeline = pipeline
        self._model_name = model_name

    def name(self) -> str:
        """
        Returns the identifier of the model.

        Returns:
            str: The model name.
        """
        return self._model_name

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the underlying Scikit-learn pipeline using the provided data.

        Args:
            x (np.ndarray): Feature matrix or EEG epochs of shape (n_samples, ...).
            y (np.ndarray): Target labels for the training samples.
        """
        self._pipeline.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input samples using the trained pipeline.

        Args:
            x (np.ndarray): Input samples to classify.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self._pipeline.predict(x)

    def predict_class_probability(self, x: np.ndarray) -> np.ndarray:
        """
        Estimates class probabilities for the input samples.

        Args:
            x (np.ndarray): Input samples for which to estimate probabilities.

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).

        Raises:
            AttributeError: If the underlying pipeline does not support probability estimation.
        """
        if hasattr(self._pipeline, "predict_proba"):
            return self._pipeline.predict_proba(x)
        raise AttributeError(f"Model {self._model_name} does not have a predict_proba method.")

    def get_state_dict(self) -> dict:
        """
        Extracts the serializable state of the model.

        Returns:
            dict: A dictionary containing the internal Scikit-learn pipeline,
                suitable for use with the SklearnModelSerializer.
        """
        return {"pipeline": self._pipeline}
