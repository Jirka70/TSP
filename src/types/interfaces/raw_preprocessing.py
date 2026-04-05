from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO


class IRawPreprocessing(ABC):
    """An interface for the raw preprocessing step in the pipeline."""

    @abstractmethod
    def run(self, input_dto: RawPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[RawPreprocessedDTO]:
        """
        Runs the raw preprocessing step in the pipeline.

        Args:
            input_dto (RawPreprocessingInputDTO): The raw preprocessing DTO.
            run_ctx (RunContext): The context to run the raw preprocessing step.
        """
        raise NotImplementedError
