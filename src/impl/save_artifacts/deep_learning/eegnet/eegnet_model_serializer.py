from pathlib import Path

import numpy
import torch
from numpy.typing import NDArray

from src.impl.model.deep_learning.eegnet.load.eegnet_model_loader import EEGNetModelLoader
from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.impl.save_artifacts.deep_learning.eegnet.serialization_input_schema import ValidatedEEGNetSerializationInput
from src.pipeline.context.run_context import RunContext
from src.types.dto.model.deep_learning.eegnet.checkpoint_format import EEGNET_CHECKPOINT_FORMAT, EEGNET_FORMAT_VERSION
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.save_artifacts.artifact_ref import ArtifactRef
from src.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO
from src.types.interfaces.model.model_serializer import IModelSerializer

EEGNET_MODEL_NAME = "eegnet"


class EEGNetModelSerializer(IModelSerializer):
    EEGNET_CHECKPOINT_FORMAT = EEGNET_CHECKPOINT_FORMAT
    EEGNET_FORMAT_VERSION = EEGNET_FORMAT_VERSION

    def supports(self, model_name: str) -> bool:
        return model_name == EEGNET_MODEL_NAME

    def save(
        self,
        trained_model: TrainedModelDTO,
        output_path: Path,
        run_ctx: RunContext | None = None,
    ) -> SavedArtifactsDTO:
        validated_input = ValidatedEEGNetSerializationInput.model_validate(
            {
                "model": trained_model.model,
                "verification_data": trained_model.verification_data,
                "run_context": run_ctx,
            }
        )
        verification_dataset = validated_input.verification_data

        expected_labels = validated_input.model.predict(verification_dataset.x)
        expected_probabilities = validated_input.model.predict_class_probability(verification_dataset.x)
        saved_artifacts = self._save_checkpoint(trained_model, output_path, validated_input.model)

        (model_artifact,) = saved_artifacts.artifacts
        artifact_directory = model_artifact.path.parent.parent
        loaded_model = EEGNetModelLoader().load(artifact_directory, validated_input.run_context)
        actual_labels = loaded_model.predict(verification_dataset.x)
        actual_probabilities = loaded_model.predict_class_probability(verification_dataset.x)

        if not _are_predicted_labels_equal(expected_labels, actual_labels):
            raise ValueError("Predicted labels changed after saving and loading EEGNet.")
        if not _are_probabilities_close(expected_probabilities, actual_probabilities):
            raise ValueError("Probabilities changed after saving and loading EEGNet.")
        return saved_artifacts

    def _save_checkpoint(
        self,
        trained_model: TrainedModelDTO,
        output_path: Path,
        model: EEGNetModel,
    ) -> SavedArtifactsDTO:
        output_path.mkdir(parents=True, exist_ok=True)

        safe_model_name = trained_model.model_name.replace(" ", "_").lower()
        file_path = output_path / f"{safe_model_name}.pt"

        checkpoint = {
            "format": self.EEGNET_CHECKPOINT_FORMAT,
            "format_version": self.EEGNET_FORMAT_VERSION,
            "model_name": trained_model.model_name,
            "model_state": {**model.get_state_dict(), "input_shape": model.input_shape},
            "best_epoch": model.best_epoch,
            "best_validation_metric_name": trained_model.best_validation_metric_name,
            "best_validation_metric_value": trained_model.best_validation_metric_value,
            "metadata": trained_model.metadata,
        }

        torch.save(checkpoint, file_path)

        return SavedArtifactsDTO(
            artifacts=[
                ArtifactRef(
                    name="trained_model",
                    path=file_path,
                    kind="model",
                    metadata={
                        "format": self.EEGNET_CHECKPOINT_FORMAT,
                        "format_version": self.EEGNET_FORMAT_VERSION,
                        "model_name": trained_model.model_name,
                    },
                )
            ]
        )


def _are_predicted_labels_equal(expected: NDArray[numpy.generic], actual: NDArray[numpy.generic]) -> bool:
    return bool(numpy.array_equal(expected, actual))


def _are_probabilities_close(expected: NDArray[numpy.generic], actual: NDArray[numpy.generic]) -> bool:
    if expected.shape != actual.shape:
        return False
    return bool(numpy.allclose(expected, actual, rtol=1e-5, atol=1e-6, equal_nan=False))
