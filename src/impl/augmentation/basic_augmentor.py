"""Basic data augmentation module for EEG data."""

import logging

import numpy as np

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.interfaces.augmentor import IAugmentor


class BasicAugmentor(IAugmentor):
    """Augmentor applying basic transformations like noise, shift, and dropout."""

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[EpochPreprocessedDTO]:
        """Execute basic augmentation on the input data."""
        log = logging.getLogger(__name__)
        config = input_dto.augmentationConfig

        # Check if augmentation is enabled
        if not config.enabled:
            log.info("Augmentation is disabled in config. Skipping.")
            return StepResult(input_dto.epoch_data)

        log.info(f"Running BasicAugmentor: {config.copies_per_sample} copies per sample")

        x = input_dto.epoch_data.signal
        original_labels = input_dto.epoch_data.labels

        # Initialize lists with original data
        augmented_x_list = [x]
        augmented_labels = [original_labels]

        # Main augmentation loop
        for _ in range(config.copies_per_sample):
            x_aug = np.copy(x)

            # Add Gaussian noise
            if config.gaussian_noise_std > 0.0:
                noise = np.random.normal(0, config.gaussian_noise_std, size=x_aug.shape)
                x_aug += noise

            # Apply time shift
            if config.max_time_shift > 0:
                shifts = np.random.randint(
                    -config.max_time_shift,
                    config.max_time_shift + 1,
                    size=x_aug.shape[0],
                )
                for i, shift in enumerate(shifts):
                    x_aug[i] = np.roll(x_aug[i], shift, axis=-1)

            # Apply channel dropout
            if config.channel_dropout_prob > 0.0:
                mask_shape = (x_aug.shape[0], x_aug.shape[1], 1)
                dropout_mask = (np.random.rand(*mask_shape) > config.channel_dropout_prob).astype(np.float32)
                x_aug = x_aug * dropout_mask

            augmented_x_list.append(x_aug)
            augmented_labels.append(original_labels)

        # Combine into final arrays
        final_signal = np.vstack(augmented_x_list)
        final_labels = np.concatenate(augmented_labels)

        log.info(f"Augmentation complete. Expanded epochs from {x.shape[0]} to {final_signal.shape[0]}")

        return StepResult(EpochPreprocessedDTO(signal=final_signal, labels=final_labels))
