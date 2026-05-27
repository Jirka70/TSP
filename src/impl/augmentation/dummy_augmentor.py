from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.augmentor import IAugmentor


class DummyAugmentor(IAugmentor):
    """
    A pass-through augmentor for the pipeline.
    Ensures data moves to the next step without modifications.
    """

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Executes a dummy augmentation step that makes no changes.

        This method acts as a placeholder when no augmentation is required.

        Args:
            input_dto (AugmentationInputDTO): Input object containing the
                dataset splits and augmentation configuration.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[DatasetSplitDTO]: A step result wrapping the unchanged
                dataset splits.
        """
        log = run_ctx.logger.for_step("DUMMY_AUGMENTATION")
        log.info("Running dummy augmentor")
        return StepResult(input_dto.data)
