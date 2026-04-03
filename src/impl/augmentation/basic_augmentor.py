"""Basic data augmentation module for EEG data."""

import logging

import numpy as np

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.interfaces.augmentor import EpochingDataDTO, IAugmentor


class BasicAugmentor(IAugmentor):
    """Augmentor applying basic transformations like noise, shift, and dropout."""

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[EpochingDataDTO]:
        """Execute basic augmentation on the input data."""
        log = logging.getLogger(__name__)
        config = input_dto.augmentationConfig

        # Check if augmentation is enabled
        if not config.enabled:
            log.info("Augmentation is disabled in config. Skipping.")
            return StepResult(input_dto.epoch_data)

        log.info(f"Running BasicAugmentor: {config.copies_per_sample} copies per sample")

        # Extract raw data to numpy array
        raw_data = input_dto.epoch_data.data
        if hasattr(raw_data, "get_data"):
            x = raw_data.get_data()
        else:
            x = np.array(raw_data)

        original_labels = input_dto.epoch_data.labels
        original_events = input_dto.epoch_data.event_names

        # Initialize lists with original data
        augmented_x_list = [x]
        augmented_labels = list(original_labels)
        augmented_events = list(original_events)

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
                    x_aug[i] = np.roll(x_aug[i], shift, axis=1)

            # Apply channel dropout
            if config.channel_dropout_prob > 0.0:
                mask_shape = (x_aug.shape[0], x_aug.shape[1], 1)
                dropout_mask = (np.random.rand(*mask_shape) > config.channel_dropout_prob).astype(np.float32)
                x_aug = x_aug * dropout_mask

            augmented_x_list.append(x_aug)
            augmented_labels.extend(original_labels)
            augmented_events.extend(original_events)

        # Combine into final array
        final_data = np.vstack(augmented_x_list)
        new_n_epochs = final_data.shape[0]

        log.info(f"Augmentation complete. Expanded epochs from {x.shape[0]} to {new_n_epochs}")

        # Pack back to DTO
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
