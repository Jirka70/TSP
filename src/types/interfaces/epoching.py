# src/types/interfaces/paradigm/i_epoching.py

from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.epoching.epoching_input_dto import EpochingInputDTO


class IEpoching(ABC):
    @abstractmethod
    def run(
        self,
        input_dto: EpochingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EpochingDataDTO]:
        raise NotImplementedError
