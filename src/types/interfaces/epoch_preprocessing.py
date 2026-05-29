from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO


class IEpochPreprocessing(ABC):
    """An interface for the epoch preprocessing step in the pipeline."""

    @abstractmethod
    def run(self, input_dto: EpochPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[EpochPreprocessedDTO]:
        """
        Runs the paradigm step in the pipeline.

        Args:
            input_dto (EpochPreprocessingInputDTO): The epoch preprocessing DTO.
            run_ctx (RunContext): The context to run the epoch preprocessing step.
        """
        raise NotImplementedError
