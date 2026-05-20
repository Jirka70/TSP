import logging

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from tsp_eeg_classification.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from tsp_eeg_classification.types.interfaces.evaluator import IEvaluator


class DummyEvaluator(IEvaluator):
    def run(
        self, input_dto: EvaluationInputDTO, run_ctx: RunContext
    ) -> StepResult[EvaluationResultDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy evaluator")
        result: EvaluationResultDTO = EvaluationResultDTO({})
        return StepResult(result)
