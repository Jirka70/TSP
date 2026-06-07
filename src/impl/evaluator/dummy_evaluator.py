import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator


class DummyEvaluator(IEvaluator):
    """
    A minimal placeholder evaluator for pipeline testing and verification.

    This evaluator returns empty metrics and performs no actual classification logic.
    It is useful for validating the pipeline's structural integrity.
    """

    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> StepResult[EvaluationResultDTO]:
        """
        Executes a placeholder evaluation step.

        The process follows these minimal stages:
        1. Initialization: Identifies the current step in the pipeline.
        2. Result Construction: Creates an empty EvaluationResultDTO.

        Args:
            input_dto (EvaluationInputDTO): DTO containing models and split data.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[EvaluationResultDTO]: A step result containing empty evaluation metrics.
        """
        log = run_ctx.logger.for_step("DUMMY_EVALUATION")

        # --- 1. Initialization ---
        log.info("Running dummy evaluator", VerbosityLevel.QUIET)

        # --- 2. Result Construction ---
        result: EvaluationResultDTO = EvaluationResultDTO(metrics={})
        return StepResult(result)
