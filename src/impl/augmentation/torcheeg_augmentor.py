import random

import mne
import numpy as np
import torch
from torcheeg import transforms

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.augmentation_config import AugmentationConfigTorchEEG
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.interfaces.augmentor import IAugmentor


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
            factor = random.uniform(self.min_scale, self.max_scale)
            eeg = eeg * factor

        return dict(eeg=eeg, **kwargs)


class TorchEEGAugmentor(IAugmentor):
    """
    Data augmentation pipeline using the torcheeg library.

    Supports two modes driven by the contents of :class:`DatasetSplitDTO`:

    - **Training pipeline** (``folds=None``): augments ``train`` data only;
      ``test`` and ``validation`` are passed through unchanged.
    - **Evaluation pipeline** (``folds`` is set): augments the ``train_data``
      inside each fold; fold ``test_data`` is passed through unchanged.
    """

    def run(self, input_dto: AugmentationInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Executes TorchEEG augmentation steps on epoched neural data.

        Args:
            input_dto (AugmentationInputDTO): Input object containing the
                dataset splits and augmentation configuration.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[DatasetSplitDTO]: Updated dataset splits with training data augmented.
        """
        log = run_ctx.logger.for_step("TORCHEEG_AUGMENTATION")
        config: AugmentationConfigTorchEEG = input_dto.augmentation_config
        dataset_splits: DatasetSplitDTO = input_dto.data

        # 1. Check if augmentation is enabled
        if not config.enabled:
            log.info("TorchEEG Augmentation is disabled. Passing data through unchanged.", VerbosityLevel.QUIET)
            return StepResult(dataset_splits)

        # 2. Initialize random seeds for all relevant libraries to ensure reproducibility
        log.info(f"Using random seed: {config.random_seed}", VerbosityLevel.TRACE)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # 3. Construct the TorchEEG transformation pipeline based on the config
        torcheeg_transform = self._build_transforms(config, log)
        if torcheeg_transform is not None:
            log.info(f"Applying augmentation steps: {', '.join([str(t) for t in torcheeg_transform.transforms])}", VerbosityLevel.NORMAL)
        else:
            log.warning("TorchEEG augmentation is enabled, but no transforms were configured. Augmentation stopped and returned original data.")
            return StepResult(dataset_splits)

        # --- Training pipeline mode: folds=None, augment train directly ---
        if dataset_splits.folds is None:
            if dataset_splits.train is None:
                log.warning("TorchEEGAugmentor (training mode): no train data available, passing through unchanged.")
                return StepResult(dataset_splits)

            log.info(f"Running TorchEEGAugmentor (training mode): creating {config.copies_per_sample} copies per sample.", VerbosityLevel.QUIET)
            augmented_train = self._augment_single_fold(
                train_data_dto=dataset_splits.train,
                torcheeg_transform=torcheeg_transform,
                copies_per_sample=config.copies_per_sample,
                log=log,
            )
            log.info("TorchEEG augmentation (training mode) completed.", VerbosityLevel.QUIET)
            return StepResult(DatasetSplitDTO(
                folds=None,
                train=augmented_train,
                test=dataset_splits.test,
                validation=dataset_splits.validation,
            ))

        # --- Evaluation pipeline mode: augment each fold's train_data ---
        log.info(f"Running TorchEEGAugmentor on {len(dataset_splits.folds)} fold(s): creating {config.copies_per_sample} copies per sample.", VerbosityLevel.QUIET)
        augmented_folds = []
        for fold in dataset_splits.folds:
            log.info(f"Augmenting Fold {fold.fold_idx}.", VerbosityLevel.DETAILED)
            augmented_train_data = self._augment_single_fold(
                train_data_dto=fold.train_data,
                torcheeg_transform=torcheeg_transform,
                copies_per_sample=config.copies_per_sample,
                log=log,
            )
            new_fold = FoldDTO(fold_idx=fold.fold_idx, train_data=augmented_train_data, test_data=fold.test_data)
            augmented_folds.append(new_fold)

        log.info(f"TorchEEG augmentation completed. Total folds processed: {len(augmented_folds)}", VerbosityLevel.QUIET)
        return StepResult(DatasetSplitDTO(folds=augmented_folds))

    def _build_transforms(self, config: AugmentationConfigTorchEEG, log: PipelineLogger) -> transforms.Compose | None:
        """Helper method to construct the TorchEEG transformation pipeline."""
        transform_list = []

        # Add Random Noise
        if config.gaussian_noise.std > 0.0:
            log.info(f"Gaussian noise enabled with std: {config.gaussian_noise.std}", VerbosityLevel.TRACE)
            transform_list.append(transforms.RandomNoise(p=1.0, std=config.gaussian_noise.std))

        # Add Random Masking (zeroing out segments)
        if config.mask.prob > 0.0:
            log.info(f"Mask probability enabled with value: {config.mask.prob}", VerbosityLevel.TRACE)
            transform_list.append(transforms.RandomMask(p=config.mask.prob, ratio=config.mask.ratio))

        # Add Random Shift (cyclic shift in time)
        if config.shift.prob > 0.0:
            log.info(f"Shift propability enabled with value: {config.shift.prob}", VerbosityLevel.TRACE)
            transform_list.append(transforms.RandomShift(p=config.shift.prob))

        # Add Random Sign Flip
        if config.sign_flip.prob > 0.0:
            log.info(f"Sign flip probability enabled with value: {config.sign_flip.prob}", VerbosityLevel.TRACE)
            transform_list.append(transforms.RandomSignFlip(p=config.sign_flip.prob))

        # Add Custom Amplitude Scaling
        if config.scale.prob > 0.0:
            log.info(f"Amplitude scale probability enabled with value: {config.scale.prob}", VerbosityLevel.TRACE)
            transform_list.append(RandomAmplitudeScale(p=config.scale.prob, min_scale=config.scale.min, max_scale=config.scale.max))

        return transforms.Compose(transform_list) if transform_list else None

    def _augment_single_fold(
        self,
        train_data_dto: EpochPreprocessedDTO,
        torcheeg_transform: transforms.Compose,
        copies_per_sample: int,
        log: PipelineLogger,
    ) -> EpochPreprocessedDTO:
        """Core logic for augmenting all recordings within a single training partition."""
        from src.types.dto.load.recording import RecordingDTO

        augmented_recordings = []

        total_original_samples = 0
        total_augmented_samples = 0

        for rec in train_data_dto.data:
            x_np = rec.data
            labels = None
            metadata = rec.metadata.copy() if isinstance(rec.metadata, dict) else {}

            # 1. Extract labels and preserve essential MNE metadata
            if isinstance(x_np, mne.Epochs):
                labels = x_np.events[:, -1]
                metadata.setdefault("sfreq", float(x_np.info["sfreq"]))
                metadata.setdefault("ch_names", list(x_np.ch_names))
                x_np = x_np.get_data(copy=False)
            elif "labels" in metadata:
                labels = np.array(metadata["labels"])
            else:
                labels = np.zeros(len(x_np), dtype=int)

            original_shape = x_np.shape
            n_orig = original_shape[0]
            total_original_samples += n_orig

            # Initialize lists with original data
            augmented_x_list = [x_np]
            augmented_labels_list = [labels]

            # 2. Generation loop: Vectorized batch transformation
            batch_tensor = torch.tensor(x_np, dtype=torch.float32)
            for _ in range(copies_per_sample):
                transformed = torcheeg_transform(eeg=batch_tensor.clone())["eeg"]
                augmented_x_list.append(transformed.detach().cpu().numpy())
                augmented_labels_list.append(labels)

            # 3. Concatenate all original and augmented copies
            final_signal = np.vstack(augmented_x_list)
            final_labels = np.concatenate(augmented_labels_list)
            n_total = final_signal.shape[0]
            total_augmented_samples += n_total

            # 4. Replicate any per-epoch metadata to match the new sample count
            final_metadata = metadata.copy()
            multiplier = 1 + copies_per_sample
            for k, v in metadata.items():
                if k == "labels":
                    continue
                if isinstance(v, (list, np.ndarray)) and len(v) == n_orig:
                    if isinstance(v, np.ndarray):
                        final_metadata[k] = np.tile(v, multiplier)
                    else:
                        final_metadata[k] = v * multiplier

            # 5. Shuffle combined samples so originals and copies are evenly mixed
            perm = np.random.permutation(n_total)
            final_signal = final_signal[perm]
            final_labels = final_labels[perm]
            final_metadata["labels"] = final_labels

            for k, v in final_metadata.items():
                if k == "labels":
                    continue
                if isinstance(v, np.ndarray) and len(v) == n_total:
                    final_metadata[k] = v[perm]
                elif isinstance(v, list) and len(v) == n_total:
                    final_metadata[k] = [v[p] for p in perm]

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
            f"TorchEEG fold augmentation summary: {total_original_samples} -> {total_augmented_samples} samples.",
            VerbosityLevel.DETAILED,
        )
        return EpochPreprocessedDTO(data=augmented_recordings)
