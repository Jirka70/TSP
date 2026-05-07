import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.raw_augmentation.raw_augmentation_input_dto import RawAugmentationInputDTO
from src.types.dto.raw_augmentation.raw_augmented_dto import RawAugmentedDTO
from src.types.interfaces.raw_augmentor import IRawAugmentor

log = logging.getLogger(__name__)


class DummyRawAugmentor(IRawAugmentor):
    """
    Pass-through implementation of raw augmentation.
    """

    def run(self, input_dto: RawAugmentationInputDTO, run_ctx: RunContext) -> StepResult[RawAugmentedDTO]:
        log.info(f"Using Dummy Raw Augmentor. Passing through {len(input_dto.data.data)} recordings unchanged.")
        return StepResult(RawAugmentedDTO(data=input_dto.data.data))
