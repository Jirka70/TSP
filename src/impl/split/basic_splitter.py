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

    def extract_data(self, recordings: list) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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

            # 1. Attempt to extract labels from metadata
            if isinstance(rec.metadata, dict) and "labels" in rec.metadata:
                labels = np.array(rec.metadata["labels"])

            # 2. Extract labels and data from MNE Epochs
            elif isinstance(signal, mne.BaseEpochs):
                labels = signal.events[:, -1]
                signal = signal.get_data()

            # Final safety check for labels
            if labels is None:
                raise ValueError(f"No labels found for recording {rec.subject_id}. BasicSplitter requires labels for partitioning.")

            all_signals.append(signal)
            all_labels.append(labels)

            # Replicate metadata to match the number of epochs
            # Standardize required columns for grouping if available
            subj = rec.metadata.get("subject", rec.subject_id) if isinstance(rec.metadata, dict) else rec.subject_id
            sess = rec.metadata.get("session", rec.session_id) if isinstance(rec.metadata, dict) else rec.session_id
            run = rec.metadata.get("run", rec.run_id) if isinstance(rec.metadata, dict) else rec.run_id

            row_data = {"subject": subj, "session": sess, "run": run}
            if isinstance(rec.metadata, dict):
                for k, v in rec.metadata.items():
                    if k not in ["subject", "session", "run", "labels"] and np.isscalar(v):
                        row_data[k] = v

            df_meta = pd.DataFrame([row_data] * len(labels))
            all_metadata.append(df_meta)

        # Concatenate all extracted components
        combined_signal = np.concatenate(all_signals, axis=0) if all_signals else np.array([])
        combined_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        combined_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else None

        return combined_signal, combined_labels, combined_metadata

    def create_dto(
        self,
        idx_list: np.ndarray,
        subset_x: np.ndarray,
        subset_y: np.ndarray,
        subset_metadata: pd.DataFrame,
        dataset_name: str,
    ) -> EpochPreprocessedDTO | None:
        """
        Reconstructs RecordingDTOs from aggregated data.
        """
        if idx_list is None or len(idx_list) == 0:
            return None

        from src.types.dto.load.recording import RecordingDTO

        current_metadata = subset_metadata.iloc[idx_list]
        reconstructed_recs = []

        # Group by subject, session and run to recreate original recording structure
        group_cols = ["subject", "session"]
        if "run" in current_metadata.columns:
            group_cols.append("run")

        groups = current_metadata.groupby(group_cols)

        for group_keys, group_df in groups:
            if len(group_cols) == 1:
                subj, sess, run = group_keys, "session_0", "run_0"
            elif len(group_cols) == 2:
                subj, sess = group_keys
                run = "run_0"
            else:
                subj, sess, run = group_keys

            abs_indices = group_df.index.values
            first_row = group_df.iloc[0].to_dict()
            meta_dict = {k: v for k, v in first_row.items() if k not in group_cols}
            meta_dict["labels"] = subset_y[abs_indices]

            reconstructed_recs.append(
                RecordingDTO(
                    data=subset_x[abs_indices],
                    dataset_name=dataset_name,
                    subject_id=subj,
                    session_id=str(sess),
                    run_id=run,
                    metadata=meta_dict,
                )
            )

        return EpochPreprocessedDTO(data=reconstructed_recs)

    def run(self, input_dto: SplitInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Executes the percentage-based splitting logic.
        """
        config = input_dto.config
        recordings = input_dto.data.data
        dataset_name = recordings[0].dataset_name if recordings else "unknown"

        if not config.enabled:
            log.info("BasicSplitter is disabled. Passing entire dataset as training data.")
            single_fold = FoldDTO(
                fold_idx=0,
                train_data=input_dto.data,
                test_data=None,
            )
            return StepResult(DatasetSplitDTO(folds=[single_fold], validation_data=None))

        log.info(f"Running BasicSplitter (train: {config.train_ratio}, val: {config.validation_ratio}, test: {config.test_ratio})")

        # Aggregate data from all input recordings
        x, y, metadata = self.extract_data(recordings)

        if metadata is None:
            raise ValueError("Metadata aggregation failed or no recordings provided.")

        n_samples = len(y)
        indices = np.arange(n_samples)

        if config.shuffle:
            np.random.seed(config.random_seed)
            np.random.shuffle(indices)

        validation_data_global = None
        main_indices = indices

        # 1. Pre-split validation
        if config.validation_ratio > 0 and config.pre_split_validation:
            num_val = int(n_samples * config.validation_ratio)
            if num_val > 0:
                val_indices = indices[:num_val]
                main_indices = indices[num_val:]
                validation_data_global = self.create_dto(val_indices, x, y, metadata, dataset_name)
                log.info(f"Pre-split: extracted {num_val} samples for global validation.")

        # Calculate boundaries for the main split (train/test) within main_indices
        n_main = len(main_indices)
        total_tt = config.train_ratio + config.test_ratio
        if total_tt == 0:
            raise ValueError("Both train_ratio and test_ratio are zero.")

        train_share = config.train_ratio / total_tt
        num_train_all = int(n_main * train_share)

        train_idx_all = main_indices[:num_train_all]
        test_idx = main_indices[num_train_all:]

        actual_train_idx = train_idx_all

        # 2. Post-split validation
        if config.validation_ratio > 0 and not config.pre_split_validation:
            num_val = int(len(train_idx_all) * config.validation_ratio)
            if num_val > 0:
                val_idx_in_fold = train_idx_all[:num_val]
                actual_train_idx = train_idx_all[num_val:]
                validation_data_global = self.create_dto(val_idx_in_fold, x, y, metadata, dataset_name)
                log.info(f"Post-split: extracted {num_val} samples from training set for validation.")

        # Create DTOs
        train_dto = self.create_dto(actual_train_idx, x, y, metadata, dataset_name)
        test_dto = self.create_dto(test_idx, x, y, metadata, dataset_name)

        log.info(f"Split completed: {len(actual_train_idx)} train, {len(test_idx)} test samples.")

        # Package results into a single fold
        single_fold = FoldDTO(
            fold_idx=0,
            train_data=train_dto,
            test_data=test_dto,
        )

        return StepResult(DatasetSplitDTO(folds=[single_fold], validation_data=validation_data_global))
