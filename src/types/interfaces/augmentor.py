from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.augmentation.augmented_samples_dto import AugmentedSamplesDTO


class EpochingDataDTO:
    pass


class IAugmentor(ABC):
    @abstractmethod
    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[EpochingDataDTO]:
        raise NotImplementedError
