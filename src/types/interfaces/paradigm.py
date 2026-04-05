from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO


class IParadigm(ABC):
    """An interface for the paradigm step in the pipeline."""

    @abstractmethod
    def run(self, input_dto: ParadigmInputDTO, run_ctx: RunContext) -> StepResult[ParadigmResultDTO]:
        """
        Runs the paradigm step in the pipeline.

        Args:
            input_dto (ParadigmInputDTO): The paradigm DTO.
            run_ctx (RunContext): The context to run the paradigm step.
        """
        raise NotImplementedError
