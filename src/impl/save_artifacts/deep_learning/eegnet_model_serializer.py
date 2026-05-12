from pathlib import Path

import torch

from src.impl.model.deep_learning.eegnet_model import EEGNetModel
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.save_artifacts.artifact_ref import ArtifactRef
from src.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO
from src.types.interfaces.model.model_serializer import IModelSerializer


class EEGNetModelSerializer(IModelSerializer):
    def supports(self, model_name: str) -> bool:
        return True

    def save(self, trained_model: TrainedModelDTO, output_path: Path) -> SavedArtifactsDTO:
        if not isinstance(trained_model.model, EEGNetModel):
            raise TypeError(
                f"EEGNetModelSerializer expected EEGNetModel, "
                f"got {type(trained_model.model).__name__}"
            )

        output_path.mkdir(parents=True, exist_ok=True)

        safe_model_name = trained_model.model_name.replace(" ", "_").lower()
        file_path = output_path / f"{safe_model_name}.pt"

        model = trained_model.model
        checkpoint = {
            "format": "eegnet_checkpoint",
            "model_name": trained_model.model_name,
            "model_state": model.get_state_dict(),
            "history": model.history,
            #"best_epoch": model.best_epoch,
            #"best_validation_metric_name": trained_model.best_validation_metric_name,
            #"best_validation_metric_value": trained_model.best_validation_metric_value,
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
                        "format": "eegnet_checkpoint",
                        "model_name": trained_model.model_name
                    }
                )
            ]
        )