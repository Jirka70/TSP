import logging
import random

import mne
import numpy as np
import torch
from torcheeg import transforms

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.raw_augmentation_config import RawAugmentationConfigTorchEEG
from src.types.dto.raw_augmentation.raw_augmentation_input_dto import RawAugmentationInputDTO
from src.types.dto.raw_augmentation.raw_augmented_dto import RawAugmentedDTO
from src.types.interfaces.raw_augmentor import IRawAugmentor

log = logging.getLogger(__name__)


class RandomAmplitudeScale:
    """
    Custom TorchEEG compatible transform for amplitude scaling.
    Multiplies the EEG signal by a random factor within a specified range.
    """

    def __init__(self, p: float, min_scale: float, max_scale: float):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, eeg: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        if random.random() < self.p:
            factor = random.uniform(self.min_scale, self.max_scale)
            eeg = eeg * factor
        return dict(eeg=eeg, **kwargs)


class TorchEEGRawAugmentor(IRawAugmentor):
    """
    Performs raw signal augmentation using the torcheeg library.
    Applies transformations such as Gaussian noise addition, random
    masking, sign flipping, and amplitude scaling based on the provided
    configuration. Each recording is augmented once (1-to-1 augmentation).
    """

    def run(self, input_dto: RawAugmentationInputDTO, run_ctx: RunContext) -> StepResult[RawAugmentedDTO]:
        config: RawAugmentationConfigTorchEEG = input_dto.config
        raw_preprocessed = input_dto.data

        # 1. Check if augmentation is enabled
        if not config.enabled:
            log.info("Raw TorchEEG Augmentation is disabled. Passing data through unchanged.")
            return StepResult(RawAugmentedDTO(data=raw_preprocessed.data))

        log.warning("Using TorchEEGRawAugmentor. Be aware that this may cause data leakage - consider using Cross subject splitter for save split. Also this may significantly increase the number of recordings and thus memory usage.")
        log.info(f"Running TorchEEGRawAugmentor. Recordings: {len(raw_preprocessed.data)}.")

        # Set seeds for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        transform = self._build_transforms(config)
        if not transform:
            log.warning("No raw transforms configured. Returning original data.")
            return StepResult(RawAugmentedDTO(data=raw_preprocessed.data))

        log.info(f"Applying TorchEEG transforms to {len(raw_preprocessed.data)} recordings.")
        augmented_recordings = []
        for i, recording in enumerate(raw_preprocessed.data):
            # Keep the original recording
            augmented_recordings.append(recording)

            # Variant B: Multi-session augmentation
            # Create additional sessions based on copies_per_sample
            raw: mne.io.Raw = recording.data
            log.info(f"Augmenting recording {i + 1}/{len(raw_preprocessed.data)} (Subject: {recording.subject_id})")
            data = raw.get_data()  # (channels, times)

            # Convert to tensor [C, T]
            base_tensor = torch.tensor(data, dtype=torch.float32)

            # Create copies_per_sample + 1 augmented variants
            # Range is 0 to copies_per_sample
            for copy_idx in range(config.copies_per_sample):
                # We use .clone() to ensure each transformation starts from the original data
                # preventing potential cumulative effects of in-place operations.
                transformed_data = transform(eeg=base_tensor.clone())["eeg"].numpy()

                # Create a new Raw object with transformed data
                raw_augmented = mne.io.RawArray(transformed_data, raw.info, verbose=False)

                # Add back annotations if any
                raw_augmented.set_annotations(raw.annotations)

                # Create new RecordingDTO with modified session_id
                # session_id will be 'iraw_augmented' where i is copy_idx
                new_recording = recording.__class__(
                    data=raw_augmented, dataset_name=recording.dataset_name, subject_id=recording.subject_id, session_id=f"{recording.session_id}_{copy_idx}raw_augmented", run_id=recording.run_id, metadata=recording.metadata.copy()
                )
                augmented_recordings.append(new_recording)

        return StepResult(RawAugmentedDTO(data=augmented_recordings))

    def _build_transforms(self, config: RawAugmentationConfigTorchEEG) -> transforms.Compose | None:
        """
        Builds a composition of TorchEEG transformations based on the provided configuration.

        This method checks for various augmentation options in the configuration (Gaussian noise,
        random masking, sign flipping, and amplitude scaling) and adds the corresponding
        transformations to a list. If any transformations are configured, they are composed
        into a single `transforms.Compose` object.

        Args:
            config (RawAugmentationConfigTorchEEG): The configuration object containing
                augmentation parameters.

        Returns:
            transforms.Compose | None: A composition of transformations if at least one is enabled,
                otherwise None.
        """
        transform_list = []

        if config.gaussian_noise_std > 0.0:
            transform_list.append(transforms.RandomNoise(p=1.0, std=config.gaussian_noise_std))

        if config.mask_prob > 0.0:
            transform_list.append(transforms.RandomMask(p=config.mask_prob, ratio=config.mask_ratio))

        if config.sign_flip_prob > 0.0:
            transform_list.append(transforms.RandomSignFlip(p=config.sign_flip_prob))

        if config.scale_prob > 0.0:
            transform_list.append(RandomAmplitudeScale(p=config.scale_prob, min_scale=config.scale_min, max_scale=config.scale_max))

        return transforms.Compose(transform_list) if transform_list else None
