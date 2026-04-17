from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO


class IFinalTrainer(ABC):
    @abstractmethod
    def run(self, input_dto: FinalTrainingInputDTO, run_ctx: RunContext) -> StepResult[FinalTrainingResultDTO]:
        raise NotImplementedError
