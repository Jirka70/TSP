import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.augmentor import IAugmentor


class DummyAugmentor(IAugmentor):
    def run(
        self, input_dto: AugmentationInputDTO, run_ctx: RunContext
    ) -> StepResult[DatasetSplitDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy augmentor")
        return StepResult(input_dto.data)
