from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.raw_augmentation.raw_augmented_dto import RawAugmentedDTO
from src.types.dto.raw_augmentation.raw_augmentation_input_dto import RawAugmentationInputDTO


class IRawAugmentor(ABC):
    """
    Interface for raw signal augmentation strategies.
    
    Raw augmentation is applied to continuous signals after preprocessing but before epoching (paradigm).
    """

    @abstractmethod
    def run(
        self, input_dto: RawAugmentationInputDTO, run_ctx: RunContext
    ) -> StepResult[RawAugmentedDTO]:
        """
        Executes the raw signal augmentation.
        
        Args:
            input_dto: Contains the raw augmentation configuration and the preprocessed raw data.
            run_ctx: Context of the current pipeline execution.
            
        Returns:
            StepResult containing the augmented (or unchanged) RawAugmentedDTO.
        """
        raise NotImplementedError
