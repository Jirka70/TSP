from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.paradigm.paradigm_preprocessed_dto import ParadigmPreprocessedDTO
from src.types.dto.paradigm.paradigm_preprocessing_input_dto import ParadigmPreprocessingInputDTO


class IParadigm(ABC):
    @abstractmethod
    def run(self, input_dto: ParadigmPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[ParadigmPreprocessedDTO]:
        raise NotImplementedError
