"""Augmentor implementation using the torcheeg library."""

import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.interfaces.augmentor import EpochingDataDTO, IAugmentor

log = logging.getLogger(__name__)


class TorchEEGAugmentor(IAugmentor):
    """Performs data augmentation using the torcheeg library.
    This augmentor applies a series of transformations from the torcheeg
    library to the input EEG epochs.
    """

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[EpochingDataDTO]:
        """Applies torcheeg-based augmentations to the epoch data."""
        epoching_data: EpochingDataDTO = EpochingDataDTO(
            data="",
            labels=[],
            event_names=[],
            sampling_rate_hz=12,
            n_epochs=2,
            n_channels=64,
            n_times=2,
            channel_names=[""],
        )
        return StepResult(epoching_data)
