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
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.interfaces.augmentor import IAugmentor

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

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[EpochPreprocessedDTO]:
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

        x_np = input_dto.epoch_data.signal
        original_labels = input_dto.epoch_data.labels

        augmented_x_list = [x_np]
        augmented_labels = [original_labels]

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
            augmented_labels.append(original_labels)

        # Concatenate and package back into a DTO
        final_signal = np.vstack(augmented_x_list)
        final_labels = np.concatenate(augmented_labels)

        log.info(
            "TorchEEG Augmentation complete. Original epochs: %d -> Total epochs: %d.",
            x_np.shape[0],
            final_signal.shape[0],
        )

        return StepResult(EpochPreprocessedDTO(signal=final_signal, labels=final_labels))
