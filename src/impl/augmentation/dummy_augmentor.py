import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.interfaces.augmentor import IAugmentor


class DummyAugmentor(IAugmentor):
    def run(
        self, input_dto: AugmentationInputDTO, run_ctx: RunContext
    ) -> StepResult[EpochPreprocessedDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy augmentor")
        return StepResult(input_dto.epoch_data)
