# src/types/interfaces/epoching/i_epoching.py

from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoching.epoching_input_dto import EpochingInputDTO


class IEpoching(ABC):


    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        input_dto: EpochingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EpochsDataDTO]:
        raise NotImplementedError