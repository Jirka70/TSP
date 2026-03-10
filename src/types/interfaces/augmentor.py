from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.augmentation.augmented_samples_dto import AugmentedSamplesDTO


class IAugmentor(ABC):
    @abstractmethod
    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> AugmentedSamplesDTO:
        raise NotImplementedError
