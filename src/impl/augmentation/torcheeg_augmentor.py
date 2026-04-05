"""Augmentor implementation using the torcheeg library."""

import logging
import random

import numpy as np
import torch
from torcheeg import transforms

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.augmentation_config import AugmentationConfigTorchEEG
from src.types.interfaces.augmentor import EpochingDataDTO, IAugmentor

log = logging.getLogger(__name__)


class RandomAmplitudeScale:
    """Custom TorchEEG compatible transform for amplitude scaling."""

    def __init__(self, p: float, min_scale: float, max_scale: float):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, eeg: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        if random.random() < self.p:
            # Apply amplitude scaling
            factor = random.uniform(self.min_scale, self.max_scale)
            eeg = eeg * factor

        return dict(eeg=eeg, **kwargs)


class TorchEEGAugmentor(IAugmentor):
    """Performs data augmentation using the torcheeg library.
    This augmentor applies a series of transformations from the torcheeg
    library to the input EEG epochs.
    """

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[EpochingDataDTO]:
        """Applies torcheeg-based augmentations to the epoch data."""
        config: AugmentationConfigTorchEEG = input_dto.augmentationConfig

        # 1. Check if augmentation is enabled
        if not config.enabled:
            log.info("TorchEEG Augmentation is disabled. Skipping.")
            return StepResult(input_dto.epoch_data)

        log.info(
            "Running TorchEEGAugmentor: creating %d copies per sample.",
            config.copies_per_sample,
        )

        raw_data = input_dto.epoch_data.data
        # Handle MNE Epochs objects which have a get_data() method
        if hasattr(raw_data, "get_data"):
            x_np = raw_data.get_data(copy=False)
        else:
            x_np = np.array(raw_data)

        # Original metadata
        original_labels = input_dto.epoch_data.labels
        original_events = input_dto.epoch_data.event_names

        augmented_x_list = [x_np]
        augmented_labels = list(original_labels)
        augmented_events = list(original_events)

        # Build the TorchEEG transformation pipeline (Compose)
        transform_list = []

        if config.gaussian_noise_std > 0.0:
            transform_list.append(transforms.RandomNoise(p=1.0, std=config.gaussian_noise_std))

        if config.mask_prob > 0.0:
            transform_list.append(transforms.RandomMask(p=config.mask_prob, ratio=config.mask_ratio))

        if config.shift_prob > 0.0:
            transform_list.append(transforms.RandomShift(p=config.shift_prob))

        if config.sign_flip_prob > 0.0:
            transform_list.append(transforms.RandomSignFlip(p=config.sign_flip_prob))

        if config.scale_prob > 0.0:
            transform_list.append(
                RandomAmplitudeScale(p=config.scale_prob, min_scale=config.scale_min, max_scale=config.scale_max)
            )

        if not transform_list:
            log.warning("TorchEEG augmentation is enabled, but no transforms were configured. No new data will be generated.")
            return StepResult(input_dto.epoch_data)

        torcheeg_transform = transforms.Compose(transform_list)

        for _ in range(config.copies_per_sample):
            epoch_augmented_list = []

            # TorchEEG transforms are typically applied to one epoch (sample) at a time.
            for i in range(x_np.shape[0]):
                # Convert a single epoch to a Torch Tensor (shape: n_channels, n_times)
                single_epoch_tensor = torch.tensor(x_np[i], dtype=torch.float32)

                transformed = torcheeg_transform(eeg=single_epoch_tensor)["eeg"]

                # Convert back to NumPy to maintain compatibility with the rest of the pipeline
                epoch_augmented_list.append(transformed.numpy())

            # Add the newly augmented batch to the output list
            augmented_x_list.append(np.array(epoch_augmented_list))
            augmented_labels.extend(original_labels)
            augmented_events.extend(original_events)

        # Concatenate and package back into a DTO
        final_data = np.vstack(augmented_x_list)
        new_n_epochs = final_data.shape[0]

        log.info(
            "TorchEEG Augmentation complete. Original epochs: %d -> Total epochs: %d.",
            x_np.shape[0],
            new_n_epochs,
        )

        epoching_data = EpochingDataDTO(
            data=final_data,
            labels=augmented_labels,
            event_names=augmented_events,
            sampling_rate_hz=input_dto.epoch_data.sampling_rate_hz,
            n_epochs=new_n_epochs,
            n_channels=input_dto.epoch_data.n_channels,
            n_times=input_dto.epoch_data.n_times,
            channel_names=input_dto.epoch_data.channel_names,
            metadata=input_dto.epoch_data.metadata,
        )

        return StepResult(epoching_data)
