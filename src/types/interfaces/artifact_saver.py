from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO


class IArtifactSaver(ABC):

    @abstractmethod
    def run(
        self,
        input_dto: SaveArtifactsInputDTO,
        run_ctx: RunContext
    ) -> StepResult[SavedArtifactsDTO]:
        raise NotImplementedError
