import logging
from pathlib import Path

from tsp_eeg_classification.types.dto.model.trained_model_dto import TrainedModelDTO
from tsp_eeg_classification.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO
from tsp_eeg_classification.types.interfaces.model.model_serializer import IModelSerializer


class PyTorchSerializer(IModelSerializer):
    def save(
        self, trained_model: TrainedModelDTO, output_path: Path
    ) -> SavedArtifactsDTO:
        log = logging.getLogger(__name__)
        log.info("Serializing trained model with " + __class__.__name__)

        return SavedArtifactsDTO([])

    def supports(self, model_name: str) -> bool:
        return model_name.lower() == "eegnet"
