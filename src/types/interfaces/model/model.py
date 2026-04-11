from abc import ABC, abstractmethod


class IModel(ABC):
    """
    Abstract base class defining a unified interface for machine and deep learning models.

    Ensures compatibility across various backends (Scikit-learn,
    Skorch, Braindecode) within the training pipeline.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns name or unique identifier of the model.

        Returns:
            str: The model's identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y) -> None:
        """
        Trains the model using the provided feature matrix and target labels.

        Args:
            x (np.ndarray or torch.Tensor): The training data (epochs or features).
            y (np.ndarray): The corresponding class labels.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """
        Performs inference and predicts class labels for the given input data.

        Args:
            x (np.ndarray or torch.Tensor): The input data to be classified.

        Returns:
            np.ndarray: An array containing the predicted class labels.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_class_probability(self, x):
        """
        Calculates the class probability estimates for the given input data.

        Args:
            x (np.ndarray or torch.Tensor): The input data for which probabilities are estimated.

        Returns:
            np.ndarray: An array of shape (n_samples, n_classes) with probability estimates.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self) -> object:
        """
        Retrieves the internal serializable state of the model.

        This method is for the ArtifactSaver to extract internal
        parameters (e.g., Scikit-learn pipelines or PyTorch weights) for storage.

        Returns:
            object: A dictionary or object representing the model's current state.
        """
        raise NotImplementedError
