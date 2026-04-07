import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO
from src.types.interfaces.artifact_saver import IArtifactSaver
from src.types.interfaces.model.model_serializer import IModelSerializer


class UnsupportedModelSerializerError(Exception):
    pass


class ArtifactSaver(IArtifactSaver):
    def run(
        self, input_dto: SaveArtifactsInputDTO, run_ctx: RunContext
    ) -> StepResult[SavedArtifactsDTO]:
        log = logging.getLogger(__name__)
        log.info("Saving trained model")
        serializer: IModelSerializer = input_dto.model_serializer
        trained_model = input_dto.trained_model
        model_name = trained_model.model_name

        if not serializer.supports(model_name):
            raise UnsupportedModelSerializerError(
                f"Model serializer {serializer.__class__.__name__} "
                f"does not support model {model_name}"
            )

        saved_artifacts: SavedArtifactsDTO = serializer.save(
            trained_model, input_dto.output_path
        )
        return StepResult(saved_artifacts)
