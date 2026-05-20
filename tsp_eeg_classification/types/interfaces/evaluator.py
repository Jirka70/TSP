from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from tsp_eeg_classification.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO


class IEvaluator(ABC):
    @abstractmethod
    def run(
        self, input_dto: EvaluationInputDTO, run_ctx: RunContext
    ) -> StepResult[EvaluationResultDTO]:
        raise NotImplementedError
