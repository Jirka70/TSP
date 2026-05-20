from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from tsp_eeg_classification.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO


class IArtifactSaver(ABC):
    @abstractmethod
    def run(
        self, input_dto: SaveArtifactsInputDTO, run_ctx: RunContext
    ) -> StepResult[SavedArtifactsDTO]:
        raise NotImplementedError
