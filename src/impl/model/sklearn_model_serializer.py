from pathlib import Path
from typing import List

import joblib

from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO
from src.types.interfaces.model.model_serializer import IModelSerializer
from src.types.dto.save_artifacts.artifact_ref import ArtifactRef


class SklearnModelSerializer(IModelSerializer):
    """
    Concrete implementation of the IModelSerializer interface for Scikit-learn models.

    This serializer specifically handles the persistence of Scikit-learn pipelines
    using the joblib library. It focuses on saving the model's internal state
    (state_dict) to a .joblib file for future inference or analysis.
    """

    def supports(self, model_name: str) -> bool:
        """
        Checks if the given model name belongs to the Scikit-learn family supported by this serializer.

        Args:
            model_name (str): The unique identifier of the model (e.g., 'csp_lda', 'riemannian_lda').

        Returns:
            bool: True if the model is a supported Scikit-learn method, False otherwise.
        """
        # Models supporter in factory
        supported_methods : List[str] = [
            "csp_lda",
            "riemannian_lda",
            "riemannian_svm",
            "rimannian_lr",
            "riamannian_rf",
            "riamannian_mdm"
        ]
        return model_name in supported_methods

    def save(self, trained_model: TrainedModelDTO, output_path: Path) -> SavedArtifactsDTO:
        """
        Serializes the trained Scikit-learn model's state to a binary file.

        The method extracts the internal state dictionary from the model object
        and persists it using joblib.dump. It ensures the destination directory
        exists before writing the file.

        Args:
            trained_model (TrainedModelDTO): DTO containing the trained model instance
                and its metadata.
            output_path (Path): The directory where the .joblib file should be created.

        Returns:
            SavedArtifactsDTO: A DTO containing the path to the generated .joblib artifact.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        file_path : Path = output_path / f"{trained_model.model_name}.joblib"

        # Serialize and save the internal state dictionary
        joblib.dump(trained_model.model.get_state_dict(), file_path)

        # Create wrapper for the saved model
        model_artifact: ArtifactRef = ArtifactRef(
            name="trained_model",
            path=file_path,
            kind="model"
        )

        return SavedArtifactsDTO(artifacts=[model_artifact])
