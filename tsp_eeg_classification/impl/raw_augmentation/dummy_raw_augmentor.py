import logging

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.raw_augmentation.raw_augmentation_input_dto import RawAugmentationInputDTO
from tsp_eeg_classification.types.dto.raw_augmentation.raw_augmented_dto import RawAugmentedDTO
from tsp_eeg_classification.types.interfaces.raw_augmentor import IRawAugmentor

log = logging.getLogger(__name__)


class DummyRawAugmentor(IRawAugmentor):
    """
    Pass-through implementation of raw augmentation.
    """

    def run(self, input_dto: RawAugmentationInputDTO, run_ctx: RunContext) -> StepResult[RawAugmentedDTO]:
        log.info(f"Using Dummy Raw Augmentor. Passing through {len(input_dto.data.data)} recordings unchanged.")
        return StepResult(RawAugmentedDTO(data=input_dto.data.data))
