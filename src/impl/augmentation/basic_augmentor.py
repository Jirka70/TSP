"""Basic data augmentation module for EEG data."""

import logging

import mne
import numpy as np

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.augmentation_config import AugmentationConfigBasic
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.interfaces.augmentor import IAugmentor

log = logging.getLogger(__name__)


class BasicAugmentor(IAugmentor):
    """Augmentor applying basic transformations like noise, shift, and dropout."""

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """Iterates over folds, applies basic augmentation to train_data, and returns updated folds."""
        config: AugmentationConfigBasic = input_dto.augmentation_config
        dataset_splits: DatasetSplitDTO = input_dto.data

        # 1. Check if augmentation is enabled
        if not config.enabled:
            log.info("Basic Augmentation is disabled. Passing data through unchanged.")
            return StepResult(dataset_splits)

        log.info(f"Running BasicAugmentor on {len(dataset_splits.folds)} fold(s): creating {config.copies_per_sample} copies per sample.")
        log.info(f"Using random seed: {config.random_seed}")
        np.random.seed(config.random_seed)
        active_augs = []
        if config.gaussian_noise_std > 0:
            active_augs.append(f"Gaussian Noise (std={config.gaussian_noise_std})")
        if config.max_time_shift > 0:
            active_augs.append(f"Time Shift (max={config.max_time_shift})")
        if config.channel_dropout_prob > 0:
            active_augs.append(f"Channel Dropout (prob={config.channel_dropout_prob})")
        if active_augs:
            log.info(f"Active augmentations: {', '.join(active_augs)}")
        else:
            log.warning("Basic Augmentation is enabled but no specific transformations are configured. Only copies will be created.")

        augmented_folds = []

        # 2. Loop over all folds
        for fold in dataset_splits.folds:
            log.info(f"Augmenting Fold {fold.fold_idx}...")

            # Augment ONLY the training data
            augmented_train_data = self._augment_single_fold(train_data_dto=fold.train_data, config=config)

            # Reconstruct the Fold with augmented training data and untouched test data
            new_fold = FoldDTO(fold_idx=fold.fold_idx, train_data=augmented_train_data, test_data=fold.test_data)
            augmented_folds.append(new_fold)

        # 3. Return wrapped in DatasetSplitDTO
        return StepResult(DatasetSplitDTO(folds=augmented_folds, validation_data=dataset_splits.validation_data))

    def _augment_single_fold(self, train_data_dto: EpochPreprocessedDTO, config: AugmentationConfigBasic) -> EpochPreprocessedDTO:
        """Core logic for augmenting a single EpochPreprocessedDTO with basic transformations."""
        from src.types.dto.load.recording import RecordingDTO

        augmented_recordings = []

        total_original_samples = 0
        total_augmented_samples = 0

        for rec in train_data_dto.data:
            x = rec.data
            labels = None

            # Handle MNE vs NumPy
            if isinstance(x, mne.Epochs):
                labels = x.events[:, -1]
                x = x.get_data(copy=False)
            elif isinstance(rec.metadata, dict) and "labels" in rec.metadata:
                labels = rec.metadata["labels"]
            else:
                labels = np.zeros(len(x))

            original_shape = x.shape
            total_original_samples += original_shape[0]

            # Initialize lists with original data
            augmented_x_list = [x]
            augmented_labels_list = [labels]

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
                    for idx, shift in enumerate(shifts):
                        x_aug[idx] = np.roll(x_aug[idx], shift, axis=-1)

                # Apply channel dropout
                if config.channel_dropout_prob > 0.0:
                    mask_shape = (x_aug.shape[0], x_aug.shape[1], 1)
                    dropout_mask = (np.random.rand(*mask_shape) > config.channel_dropout_prob).astype(np.float32)
                    x_aug = x_aug * dropout_mask

                augmented_x_list.append(x_aug)
                augmented_labels_list.append(labels)

            # Combine into final arrays
            final_signal = np.vstack(augmented_x_list)
            final_labels = np.concatenate(augmented_labels_list)
            total_augmented_samples += final_signal.shape[0]

            log.debug(f"Recording {rec.subject_id}_{rec.session_id}_{rec.run_id}: Original shape: {original_shape}, Augmented shape: {final_signal.shape}")

            # Metadata replication
            final_metadata = rec.metadata.copy()
            if "labels" in final_metadata:
                final_metadata["labels"] = final_labels

            new_rec = RecordingDTO(data=final_signal, dataset_name=rec.dataset_name, subject_id=rec.subject_id, session_id=rec.session_id, run_id=rec.run_id, metadata=final_metadata)
            augmented_recordings.append(new_rec)

        log.info(f"Fold augmentation summary: Total samples before: {total_original_samples}, After: {total_augmented_samples}")
        return EpochPreprocessedDTO(data=augmented_recordings)
