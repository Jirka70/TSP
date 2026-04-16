"""Basic splitting module for percentage-based data partitioning."""

import logging

import mne
import numpy as np
import pandas as pd

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.splitter import ISplitter

log = logging.getLogger(__name__)


class BasicSplitter(ISplitter):
    """
    Splitter that performs a simple percentage-based split (train/validation/test).

    This implementation aggregates all input recordings into a single pool and then
    partitions them according to the configured ratios. It returns a DatasetSplitDTO
    containing a single fold (Fold 0).
    """

    def _extract_data(self, recordings: list) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Helper method to extract and aggregate signals, labels, and metadata from RecordingDTOs.

        Args:
            recordings: A list of RecordingDTO objects.

        Returns:
            A tuple containing:
                - combined_signal: A NumPy array of all signal epochs.
                - combined_labels: A NumPy array of all corresponding labels.
                - combined_metadata: A Pandas DataFrame containing merged metadata.
        """
        all_signals = []
        all_labels = []
        all_metadata = []

        for rec in recordings:
            signal = rec.data
            labels = None

            # 1. Attempt to extract labels from metadata (fallback mechanism)
            if isinstance(rec.metadata, dict) and "labels" in rec.metadata:
                labels = np.array(rec.metadata["labels"])

            # 2. Extract labels and data from MNE Epochs if applicable
            elif isinstance(signal, mne.BaseEpochs):
                labels = signal.events[:, -1]
                # Extracting raw data as a NumPy array
                signal = signal.get_data()

            # 3. Debug branch: If no labels are found, log a critical error
            else:
                meta_keys = list(rec.metadata.keys()) if isinstance(rec.metadata, dict) else "Metadata is not a dict"
                log.error(f"CRITICAL ERROR: No labels found for recording {rec.subject_id}!\n  -> Signal type: {type(signal)}\n  -> Metadata keys: {meta_keys}")

            # Final safety check for labels
            if labels is None:
                log.warning(f"No labels found for recording {rec.subject_id}. Creating zero-filled array.")
                labels = np.zeros(len(signal))

            all_signals.append(signal)
            all_labels.append(labels)

            # Replicate metadata to match the number of epochs for consistent indexing
            if isinstance(rec.metadata, dict):
                df_meta = pd.DataFrame([rec.metadata] * len(labels))
                all_metadata.append(df_meta)

        # Concatenate all extracted components into unified structures
        combined_signal = np.concatenate(all_signals, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()

        return combined_signal, combined_labels, combined_metadata

    def run(self, input_dto: SplitInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Executes the percentage-based splitting logic.

        Args:
            input_dto: Contains the split configuration (ratios, shuffle, seed) and the input data.
            run_ctx: Context of the current pipeline execution.

        Returns:
            StepResult containing a single fold with partitioned train, validation, and test sets.
        """
        config = input_dto.config
        recordings = input_dto.data.data

        # If splitting is disabled, wrap the entire dataset into a single 'training' fold
        if not config.enabled:
            log.info("BasicSplitter is disabled. Passing entire dataset as training data.")
            single_fold = FoldDTO(
                fold_idx=0,
                train_data=input_dto.data,
                validation_data=None,
                test_data=None,
            )
            return StepResult(DatasetSplitDTO(folds=[single_fold]))

        log.info(f"Running BasicSplitter (train: {config.train_ratio}, val: {config.validation_ratio}, test: {config.test_ratio})")

        # Aggregate data from all input recordings
        signal, labels, metadata = self._extract_data(recordings)

        n_samples = len(labels)
        indices = np.arange(n_samples)

        # Apply shuffling if configured
        if config.shuffle:
            np.random.seed(config.random_seed)
            np.random.shuffle(indices)

        # Calculate split boundaries based on ratios
        train_end = int(n_samples * config.train_ratio)
        val_end = train_end + int(n_samples * config.validation_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        def create_dto(idx_list: np.ndarray) -> EpochPreprocessedDTO | None:
            """Helper to wrap indices into the pipeline's DTO structure."""
            if len(idx_list) == 0:
                return None

            from src.types.dto.load.recording import RecordingDTO

            # Create a single 'combined' RecordingDTO representing the subset
            subset_rec = RecordingDTO(data=signal[idx_list], dataset_name=recordings[0].dataset_name if recordings else "unknown", subject_id="combined", session_id="combined", run_id="combined", metadata={"labels": labels[idx_list]})
            return EpochPreprocessedDTO(data=[subset_rec])

        # Create DTOs for each partition
        train_dto = create_dto(train_idx)
        val_dto = create_dto(val_idx)
        test_dto = create_dto(test_idx)

        log.info(f"Split completed: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples.")

        # Package results into a single fold
        single_fold = FoldDTO(
            fold_idx=0,
            train_data=train_dto,
            validation_data=val_dto,
            test_data=test_dto,
        )

        return StepResult(DatasetSplitDTO(folds=[single_fold]))
