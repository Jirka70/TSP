from abc import ABC, abstractmethod
from pathlib import Path

from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO


class IModelSerializer(ABC):
    """
    Abstract interface for model serialization strategies.

    This interface allows the framework to handle different storage formats
    (e.g., joblib for Scikit-learn, .pt for PyTorch/Skorch) based on the
    specific requirements of the trained model.
    """

    @abstractmethod
    def supports(self, model_name: str) -> bool:
        """
        Evaluates whether the serializer is capable of handling a specific model type.

        This method is used by the ArtifactSaver to select the correct
        serialization strategy at runtime.

        Args:
            model_name (str): The name or identifier of the model (e.g., 'csp_lda', 'eegnet').

        Returns:
            bool: True if the model name is supported by this serializer, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, trained_model: TrainedModelDTO, output_path: Path) -> SavedArtifactsDTO:
        """
        Serializes the trained model and its associated metadata to the file system.

        This method handles the actual file operations, ensuring that the
        model state, weights, and configuration are persisted correctly.

        Args:
            trained_model (TrainedModelDTO): The data transfer object containing
                the trained model instance and its training history.
            output_path (Path): The directory path where the artifacts should be saved.

        Returns:
            SavedArtifactsDTO: A DTO containing paths to all generated files
                and any additional artifact metadata.
        """
        raise NotImplementedError
