from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDto


class IRawPreprocessing(ABC):
    @abstractmethod
    def run(self, input_dto: RawPreprocessingInputDto, run_ctx: RunContext) -> StepResult[RawPreprocessedDTO]:
        raise NotImplementedError
