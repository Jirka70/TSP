from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO


class IPreprocessing(ABC):
    @abstractmethod
    def run(
        self, input_dto: PreprocessingInputDTO, run_ctx: RunContext
    ) -> StepResult[PreprocessedDataDTO]:
        raise NotImplementedError
