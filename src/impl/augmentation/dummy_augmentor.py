import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.interfaces.augmentor import IAugmentor, EpochingDataDTO


class DummyAugmentor(IAugmentor):
    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[EpochingDataDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy splitter")
        epochingData: EpochingDataDTO = EpochingDataDTO(data="",
                                                        labels=[],
                                                        event_names=[],
                                                        sampling_rate_hz=12,
                                                        n_epochs=2,
                                                        n_channels=64,
                                                        n_times=2,
                                                        channel_names=[""])
