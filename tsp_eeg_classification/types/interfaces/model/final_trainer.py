from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from tsp_eeg_classification.types.dto.model.final_training_result_dto import FinalTrainingResultDTO


class IFinalTrainer(ABC):
    """Contract for final-stage trainers that fit on all available folds."""

    @abstractmethod
    def run(self, input_dto: FinalTrainingInputDTO, run_ctx: RunContext) -> StepResult[FinalTrainingResultDTO]:
        """Train a final model using all available data and return the result."""
        raise NotImplementedError
