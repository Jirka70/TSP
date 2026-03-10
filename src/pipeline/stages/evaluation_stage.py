from src.pipeline.context.run_context import RunContext
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator


class EvaluationStage:
    def __init__(self, evaluator: IEvaluator) -> None:
        self._evaluator = evaluator

    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> EvaluationResultDTO:
        return self._evaluator.run(input_dto, run_ctx)