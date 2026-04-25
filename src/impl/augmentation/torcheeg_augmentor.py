"""Augmentor implementation using the torcheeg library."""

import logging
import random

import mne
import numpy as np
import torch
from torcheeg import transforms

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.augmentation_config import AugmentationConfigTorchEEG
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.interfaces.augmentor import IAugmentor

log = logging.getLogger(__name__)


class RandomAmplitudeScale:
    """
    Custom TorchEEG compatible transform for amplitude scaling.
    Multiplies the EEG signal by a random factor within a specified range.
    """

    def __init__(self, p: float, min_scale: float, max_scale: float):
        """
        Args:
            p: Probability of applying the transformation.
            min_scale: Minimum scaling factor.
            max_scale: Maximum scaling factor.
        """
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, eeg: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """
        Applies the random scaling to the input tensor.

        Returns:
            Dictionary containing the (possibly) transformed 'eeg' tensor.
        """
        if random.random() < self.p:
            # Generate a random factor within the [min, max] range
            factor = random.uniform(self.min_scale, self.max_scale)
            eeg = eeg * factor

        return dict(eeg=eeg, **kwargs)


class TorchEEGAugmentor(IAugmentor):
    """
    Performs data augmentation using the torcheeg library.

    This augmentor applies a series of transformations from the torcheeg
    library to the input EEG epochs, iterating over all cross-validation folds.
    It generates multiple augmented copies for each original training sample.
    """

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Main entry point for the augmentation stage.
        Iterates over folds, applies augmentation only to training data, and returns updated folds.

        Args:
            input_dto: Contains the augmentation configuration and the dataset splits.
            run_ctx: Context of the current pipeline execution.

        Returns:
            StepResult containing the DatasetSplitDTO with augmented training sets.
        """
        config: AugmentationConfigTorchEEG = input_dto.augmentationConfig
        dataset_splits: DatasetSplitDTO = input_dto.data

        # 1. Check if augmentation is enabled
        if not config.enabled:
            log.info("TorchEEG Augmentation is disabled. Passing data through unchanged.")
            return StepResult(dataset_splits)

        log.info(f"Running TorchEEGAugmentor on {len(dataset_splits.folds)} fold(s): creating {config.copies_per_sample} copies per sample.")

        # 2. Initialize random seeds for all relevant libraries to ensure reproducibility
        log.info(f"Using random seed: {config.random_seed}")
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # Identify and log active transformations for better visibility
        active_augs = []
        if config.gaussian_noise_std > 0:
            active_augs.append(f"Gaussian Noise (std={config.gaussian_noise_std})")
        if config.mask_prob > 0:
            active_augs.append(f"Masking (prob={config.mask_prob}, ratio={config.mask_ratio})")
        if config.shift_prob > 0:
            active_augs.append(f"Shift (prob={config.shift_prob})")
        if config.sign_flip_prob > 0:
            active_augs.append(f"Sign Flip (prob={config.sign_flip_prob})")
        if config.scale_prob > 0:
            active_augs.append(f"Amplitude Scale (prob={config.scale_prob}, range=[{config.scale_min}, {config.scale_max}])")

        if active_augs:
            log.info(f"Active augmentations: {', '.join(active_augs)}")
        else:
            log.warning("TorchEEG Augmentation is enabled but no specific transformations are configured. Only copies will be created.")

        # 3. Construct the TorchEEG transformation pipeline based on the config
        torcheeg_transform = self._build_transforms(config)

        if not torcheeg_transform:
            log.warning("TorchEEG augmentation is enabled, but no transforms were configured. No new data will be generated.")
            return StepResult(dataset_splits)

        augmented_folds = []

        # 4. Loop over all folds (e.g., from cross-validation)
        for fold in dataset_splits.folds:
            log.info(f"Augmenting Fold {fold.fold_idx}.")

            # Augment ONLY the training data; leave validation and test sets untouched
            augmented_train_data = self._augment_single_fold(train_data_dto=fold.train_data, torcheeg_transform=torcheeg_transform, copies_per_sample=config.copies_per_sample)

            # Reconstruct the fold with the newly augmented training data
            new_fold = FoldDTO(fold_idx=fold.fold_idx, train_data=augmented_train_data, test_data=fold.test_data)
            augmented_folds.append(new_fold)

        # 5. Wrap results in StepResult and return
        return StepResult(DatasetSplitDTO(folds=augmented_folds, validation_data=dataset_splits.validation_data))

    def _build_transforms(self, config: AugmentationConfigTorchEEG) -> transforms.Compose | None:
        """
        Helper method to construct the TorchEEG transformation pipeline.

        Args:
            config: The TorchEEG augmentation configuration.

        Returns:
            A transforms.Compose object containing all configured transformations, or None if empty.
        """
        transform_list = []

        # Add Random Noise
        if config.gaussian_noise_std > 0.0:
            transform_list.append(transforms.RandomNoise(p=1.0, std=config.gaussian_noise_std))

        # Add Random Masking (zeroing out segments)
        if config.mask_prob > 0.0:
            transform_list.append(transforms.RandomMask(p=config.mask_prob, ratio=config.mask_ratio))

        # Add Random Shift (cyclic shift in time)
        if config.shift_prob > 0.0:
            transform_list.append(transforms.RandomShift(p=config.shift_prob))

        # Add Random Sign Flip
        if config.sign_flip_prob > 0.0:
            transform_list.append(transforms.RandomSignFlip(p=config.sign_flip_prob))

        # Add Custom Amplitude Scaling
        if config.scale_prob > 0.0:
            transform_list.append(RandomAmplitudeScale(p=config.scale_prob, min_scale=config.scale_min, max_scale=config.scale_max))

        return transforms.Compose(transform_list) if transform_list else None

    def _augment_single_fold(self, train_data_dto: EpochPreprocessedDTO, torcheeg_transform: transforms.Compose, copies_per_sample: int) -> EpochPreprocessedDTO:
        """
        Core logic for augmenting all recordings within a single fold.

        Args:
            train_data_dto: The training data for the fold.
            torcheeg_transform: The compiled TorchEEG transformation pipeline.
            copies_per_sample: Number of augmented copies to create for each original sample.

        Returns:
            Updated EpochPreprocessedDTO containing original and augmented recordings.
        """
        from src.types.dto.load.recording import RecordingDTO

        augmented_recordings = []

        total_original_samples = 0
        total_augmented_samples = 0

        for rec in train_data_dto.data:
            x_np = rec.data
            labels = None

            # 1. Extract labels (trying MNE Epochs first, then metadata fallback)
            if isinstance(x_np, mne.Epochs):
                labels = x_np.events[:, -1]
                x_np = x_np.get_data(copy=False)
            elif isinstance(rec.metadata, dict) and "labels" in rec.metadata:
                labels = rec.metadata["labels"]
            else:
                # Fallback if no labels are found
                labels = np.zeros(len(x_np))

            original_shape = x_np.shape
            total_original_samples += original_shape[0]

            # Initialize lists with the original data (the first copy)
            augmented_x_list = [x_np]
            augmented_labels_list = [labels]

            # 2. Generation loop: Create N copies
            for _ in range(copies_per_sample):
                epoch_augmented_list = []

                # Apply transformations to each epoch in the recording individually
                for i in range(x_np.shape[0]):
                    # Convert to tensor for TorchEEG compatibility
                    single_epoch_tensor = torch.tensor(x_np[i], dtype=torch.float32)
                    transformed = torcheeg_transform(eeg=single_epoch_tensor)["eeg"]
                    epoch_augmented_list.append(transformed.numpy())

                augmented_x_list.append(np.array(epoch_augmented_list))
                augmented_labels_list.append(labels)

            # 3. Concatenate all original and augmented copies
            final_signal = np.vstack(augmented_x_list)
            final_labels = np.concatenate(augmented_labels_list)
            total_augmented_samples += final_signal.shape[0]

            # 4. Replicate metadata and update labels
            final_metadata = rec.metadata.copy()
            if "labels" in final_metadata:
                final_metadata["labels"] = final_labels

            # Create a new RecordingDTO with the augmented data
            new_rec = RecordingDTO(data=final_signal, dataset_name=rec.dataset_name, subject_id=rec.subject_id, session_id=rec.session_id, run_id=rec.run_id, metadata=final_metadata)
            augmented_recordings.append(new_rec)

        log.info(f"Fold augmentation summary: Total samples before: {total_original_samples}, After: {total_augmented_samples}")
        return EpochPreprocessedDTO(data=augmented_recordings)
