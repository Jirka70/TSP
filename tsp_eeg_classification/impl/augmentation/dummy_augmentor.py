import logging

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO
from tsp_eeg_classification.types.interfaces.augmentor import IAugmentor


class DummyAugmentor(IAugmentor):
    def run(
        self, input_dto: AugmentationInputDTO, run_ctx: RunContext
    ) -> StepResult[DatasetSplitDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy augmentor")
        return StepResult(input_dto.data)
