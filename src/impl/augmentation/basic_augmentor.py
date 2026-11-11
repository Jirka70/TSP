import mne
import numpy as np

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.augmentation_config import AugmentationConfigBasic
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.interfaces.augmentor import IAugmentor


def _apply_linear_time_shift(epoch: np.ndarray, shift: int) -> np.ndarray:
    """
    Shifts an epoch along the time axis linearly without circular wrap-around.

    Instead of ``np.roll`` (which creates an artificial boundary step when the
    end of the epoch wraps to the beginning), missing samples at the edge are
    padded with the boundary sample value (edge mode) to preserve signal continuity.

    Args:
        epoch: 2D array of shape (n_channels, n_times).
        shift: Number of samples to shift (positive = shift right, negative = shift left).

    Returns:
        Shifted epoch of identical shape.
    """
    if shift == 0:
        return epoch

    shifted = np.empty_like(epoch)
    if shift > 0:
        # Shifted to the right: beginning is padded with the first column
        shifted[:, :shift] = epoch[:, :1]
        shifted[:, shift:] = epoch[:, :-shift]
    else:
        # Shifted to the left: end is padded with the last column
        abs_shift = -shift
        shifted[:, :-abs_shift] = epoch[:, abs_shift:]
        shifted[:, -abs_shift:] = epoch[:, -1:]
    return shifted


class BasicAugmentor(IAugmentor):
    """
    Basic augmentation pipeline for EEG data.

    Supports two modes driven by the contents of :class:`DatasetSplitDTO`:

    - **Training pipeline** (``folds=None``): augments ``train`` data only;
      ``test`` and ``validation`` are passed through unchanged.
    - **Evaluation pipeline** (``folds`` is set): augments the ``train_data``
      inside each fold; fold ``test_data`` is passed through unchanged.
    """

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Executes basic preprocessing augmentation steps on epoched neural data.

        Augmentation transformations applied to training data:
        1. Gaussian Noise: Adds random normal noise.
        2. Linear Time Shift: Shifts epochs along the time axis (non-circular, edge-padded).
        3. Channel Dropout: Randomly sets channel values to zero.

        Args:
            input_dto (AugmentationInputDTO): Input object containing the
                dataset splits and augmentation configuration.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[DatasetSplitDTO]: Updated dataset splits with training data augmented.
        """
        log = run_ctx.logger.for_step("BASIC_AUGMENTATION")
        config: AugmentationConfigBasic = input_dto.augmentation_config
        dataset_splits: DatasetSplitDTO = input_dto.data

        log.info(f"Using random seed: {config.random_seed}", VerbosityLevel.TRACE)
        np.random.seed(config.random_seed)

        active_augs = []
        if config.gaussian_noise.std > 0:
            log.info(f"Gaussian noise enabled with std: {config.gaussian_noise.std}", VerbosityLevel.TRACE)
            active_augs.append(f"Gaussian Noise (std={config.gaussian_noise.std})")
        if config.time_shift.max_samples > 0:
            log.info(f"Time shift enabled with max shift: {config.time_shift.max_samples} samples", VerbosityLevel.TRACE)
            active_augs.append(f"Time Shift (max={config.time_shift.max_samples})")
        if config.channel_dropout.prob > 0:
            log.info(f"Channel dropout enabled with probability: {config.channel_dropout.prob}", VerbosityLevel.TRACE)
            active_augs.append(f"Channel Dropout (prob={config.channel_dropout.prob})")

        if active_augs:
            log.info(f"Active augmentations: {', '.join(active_augs)}", VerbosityLevel.NORMAL)
        else:
            log.warning("Basic Augmentation has no specific transformations configured. Only copies will be created.")

        # --- Training pipeline mode: folds=None, augment train directly ---
        if dataset_splits.folds is None:
            if dataset_splits.train is None:
                log.warning("BasicAugmentor (training mode): no train data available, passing through unchanged.")
                return StepResult(dataset_splits)

            log.info(f"Running BasicAugmentor (training mode): creating {config.copies_per_sample} copies per sample.", VerbosityLevel.QUIET)
            augmented_train = self._augment_single_fold(train_data_dto=dataset_splits.train, config=config, log=log)
            log.info("Basic augmentation (training mode) completed.", VerbosityLevel.QUIET)
            return StepResult(DatasetSplitDTO(
                folds=None,
                train=augmented_train,
                test=dataset_splits.test,
                validation=dataset_splits.validation,
            ))

        # --- Evaluation pipeline mode: augment each fold's train_data ---
        log.info(f"Running BasicAugmentor on {len(dataset_splits.folds)} fold(s): creating {config.copies_per_sample} copies per sample.", VerbosityLevel.QUIET)
        augmented_folds = []
        for fold in dataset_splits.folds:
            log.info(f"Augmenting Fold {fold.fold_idx} / {len(dataset_splits.folds)}", VerbosityLevel.DETAILED)
            augmented_train_data = self._augment_single_fold(train_data_dto=fold.train_data, config=config, log=log)
            new_fold = FoldDTO(fold_idx=fold.fold_idx, train_data=augmented_train_data, test_data=fold.test_data)
            augmented_folds.append(new_fold)

        log.info(f"Basic augmentation completed. Total folds processed: {len(augmented_folds)}", VerbosityLevel.QUIET)
        return StepResult(DatasetSplitDTO(folds=augmented_folds))

    def _augment_single_fold(self, train_data_dto: EpochPreprocessedDTO, config: AugmentationConfigBasic, log: PipelineLogger) -> EpochPreprocessedDTO:
        """Core logic for augmenting a single EpochPreprocessedDTO with basic transformations."""
        from src.types.dto.load.recording import RecordingDTO

        augmented_recordings = []

        total_original_samples = 0
        total_augmented_samples = 0

        for rec in train_data_dto.data:
            x = rec.data
            labels = None
            metadata = rec.metadata.copy() if isinstance(rec.metadata, dict) else {}

            # Handle MNE vs NumPy and preserve essential metadata
            if isinstance(x, mne.Epochs):
                labels = x.events[:, -1]
                metadata.setdefault("sfreq", float(x.info["sfreq"]))
                metadata.setdefault("channel_names", list(x.ch_names))
                x = x.get_data(copy=False)
            elif "labels" in metadata:
                labels = np.array(metadata["labels"])
            else:
                labels = np.zeros(len(x), dtype=int)

            if "ch_names" in metadata and "channel_names" not in metadata:
                metadata["channel_names"] = metadata.pop("ch_names")
            elif "ch_names" in metadata:
                metadata.pop("ch_names", None)

            original_shape = x.shape
            n_orig = original_shape[0]
            total_original_samples += n_orig

            # Initialize lists with original data
            augmented_x_list = [x]
            augmented_labels_list = [labels]

            # Main augmentation loop
            for _ in range(config.copies_per_sample):
                x_aug = np.copy(x)

                # 1. Add Gaussian noise
                if config.gaussian_noise.std > 0.0:
                    noise = np.random.normal(0, config.gaussian_noise.std, size=x_aug.shape)
                    x_aug += noise

                # 2. Apply non-circular linear time shift with edge padding
                if config.time_shift.max_samples > 0:
                    shifts = np.random.randint(
                        -config.time_shift.max_samples,
                        config.time_shift.max_samples + 1,
                        size=n_orig,
                    )
                    for idx, shift in enumerate(shifts):
                        x_aug[idx] = _apply_linear_time_shift(x_aug[idx], shift)

                # 3. Apply channel dropout
                if config.channel_dropout.prob > 0.0:
                    mask_shape = (x_aug.shape[0], x_aug.shape[1], 1)
                    dropout_mask = (np.random.rand(*mask_shape) > config.channel_dropout.prob).astype(np.float32)
                    x_aug = x_aug * dropout_mask

                augmented_x_list.append(x_aug)
                augmented_labels_list.append(labels)

            # Combine original and augmented copies
            final_signal = np.vstack(augmented_x_list)
            final_labels = np.concatenate(augmented_labels_list)
            n_total = final_signal.shape[0]
            total_augmented_samples += n_total

            # Replicate any per-epoch metadata to match the new sample count
            final_metadata = metadata.copy()
            multiplier = 1 + config.copies_per_sample
            for k, v in metadata.items():
                if k in ["labels", "ch_names", "channel_names"]:
                    continue
                if isinstance(v, (list, np.ndarray)) and len(v) == n_orig:
                    if isinstance(v, np.ndarray):
                        final_metadata[k] = np.tile(v, multiplier)
                    else:
                        final_metadata[k] = v * multiplier

            # Shuffle combined samples so originals and copies are evenly mixed
            perm = np.random.permutation(n_total)
            final_signal = final_signal[perm]
            final_labels = final_labels[perm]
            final_metadata["labels"] = final_labels

            for k, v in final_metadata.items():
                if k in ["labels", "ch_names", "channel_names"]:
                    continue
                if isinstance(v, np.ndarray) and len(v) == n_total:
                    final_metadata[k] = v[perm]
                elif isinstance(v, list) and len(v) == n_total:
                    final_metadata[k] = [v[p] for p in perm]

            log.debug(
                f"Recording: {rec.subject_id}_{rec.session_id}_{rec.run_id}. "
                f"Original: {original_shape} -> Augmented: {final_signal.shape}",
                VerbosityLevel.TRACE,
            )

            new_rec = RecordingDTO(
                data=final_signal,
                dataset_name=rec.dataset_name,
                subject_id=rec.subject_id,
                session_id=rec.session_id,
                run_id=rec.run_id,
                metadata=final_metadata,
            )
            augmented_recordings.append(new_rec)

        log.info(
            f"Augmentation summary: {total_original_samples} -> {total_augmented_samples} samples.",
            VerbosityLevel.DETAILED,
        )
        return EpochPreprocessedDTO(data=augmented_recordings)
