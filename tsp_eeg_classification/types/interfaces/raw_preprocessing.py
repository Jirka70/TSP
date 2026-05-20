from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from tsp_eeg_classification.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO


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
