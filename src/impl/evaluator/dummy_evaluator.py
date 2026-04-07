import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator


class DummyEvaluator(IEvaluator):
    def run(
        self, input_dto: EvaluationInputDTO, run_ctx: RunContext
    ) -> StepResult[EvaluationResultDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy evaluator")
        result: EvaluationResultDTO = EvaluationResultDTO({})
        return StepResult(result)
