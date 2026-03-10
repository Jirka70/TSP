from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO


class IEvaluator(ABC):
    @abstractmethod
    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> EvaluationResultDTO:
        raise NotImplementedError
