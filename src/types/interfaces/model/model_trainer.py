from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO


class IModelTrainer(ABC):
    """
    Abstract interface defining the contract for model training stages.

    Implementations of this interface are responsible for taking preprocessed
    data and model configurations to produce a trained model instance ready
    for evaluation and deployment.
    """

    @abstractmethod
    def run(self, input_dto: TrainingInputDTO, run_ctx: RunContext) -> StepResult[TrainingResultDTO]:
        """
        Executes the model training process within the pipeline context.

        This method orchestrates model initialization (typically via a factory),
        data preparation, the training loop (fitting), and the generation of
        training metrics/history.

        Args:
            input_dto (TrainingInputDTO): DTO containing model configuration,
                augmented training data, and optional validation data.
            run_ctx (RunContext): Metadata and execution context for the
                current pipeline run.

        Returns:
            StepResult[TrainingResultDTO]: An object wrapping trained models
                produced for every incoming fold.
        """
        raise NotImplementedError
