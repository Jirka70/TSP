from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


class IAugmentor(ABC):
    @abstractmethod
    def run(
        self, input_dto: AugmentationInputDTO, run_ctx: RunContext
    ) -> StepResult[EpochPreprocessedDTO]:
        raise NotImplementedError
