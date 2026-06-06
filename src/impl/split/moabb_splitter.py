"""MOABB splitting module for advanced data partitioning strategies."""

import mne
import numpy as np
import pandas as pd
from hydra.utils import get_class, instantiate

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.splitter import ISplitter
from src.types.dto.config.logging.verbosity_level import VerbosityLevel


class MoabbSplitter(ISplitter):
    """
    Splitter that uses MOABB (Mother of All BCI Benchmarks) evaluation strategies.

    This splitter interfaces with MOABB's evaluation splitters (WithinSession,
    CrossSession, CrossSubject, etc.) to generate cross-validation folds that
    respect the specific structure of EEG datasets.
    """

    def extract_data(self, recordings: list[RecordingDTO]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
        """
        Extracts and aggregates signals, labels, and metadata for MOABB compatibility.

        MOABB splitters require metadata DataFrames containing specific columns
        like 'subject' and 'session' to perform their partitioning logic correctly.

        Args:
            recordings: A list of RecordingDTO objects.

        Returns:
            A tuple containing:
                - combined_signal: A NumPy array of all signal epochs.
                - combined_labels: A NumPy array of all corresponding labels.
                - combined_metadata: A Pandas DataFrame with MOABB-required metadata columns.
        """
        all_signals = []
        all_labels = []
        all_metadata = []

        for rec in recordings:
            signal = rec.data
            labels = None

            # 1. Extract labels from metadata or MNE Epochs
            if isinstance(rec.metadata, dict) and "labels" in rec.metadata:
                labels = np.array(rec.metadata["labels"])
            elif isinstance(signal, mne.BaseEpochs):
                labels = signal.events[:, -1]
                signal = signal.get_data()

            # Final safety check for labels
            if labels is None:
                raise ValueError(f"No labels found for recording {rec.subject_id}. MOABB splitters require labels for correct operation.")

            all_signals.append(signal)
            all_labels.append(labels)

            # 2. Metadata handling - MOABB requires 'subject' and 'session' columns
            # Ensure we have standardized subject, session and run identifiers
            subj = rec.metadata.get("subject", rec.subject_id) if isinstance(rec.metadata, dict) else rec.subject_id
            sess = rec.metadata.get("session", rec.session_id) if isinstance(rec.metadata, dict) else rec.session_id
            run = rec.metadata.get("run", rec.run_id) if isinstance(rec.metadata, dict) else rec.run_id

            # Create a row template for this recording's epochs
            row_data = {"subject": subj, "session": sess, "run": run}

            # Include other scalar metadata fields
            if isinstance(rec.metadata, dict):
                for k, v in rec.metadata.items():
                    if k not in ["subject", "session", "run", "labels"] and (np.isscalar(v) or k in ["ch_names", "sfreq"]):
                        row_data[k] = v

            # Replicate the row for every epoch in this recording
            df_meta = pd.DataFrame([row_data] * len(labels))
            if "ch_names" in row_data and not np.isscalar(row_data["ch_names"]):
                df_meta["ch_names"] = [row_data["ch_names"]] * len(labels)

            all_metadata.append(df_meta)

        # Merge all recordings into unified structures
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
        name_prefix: str = "combined",
    ) -> EpochPreprocessedDTO | None:
        """
        Reconstructs RecordingDTOs from aggregated data using metadata to group by subject/session/run.
        """
        if idx_list is None or len(idx_list) == 0:
            return None

        current_metadata = subset_metadata.iloc[idx_list]
        reconstructed_recs = []

        # Group by subject, session and run to recreate original recording structure
        group_cols = ["subject", "session"]
        if "run" in current_metadata.columns:
            group_cols.append("run")

        groups = current_metadata.groupby(group_cols)

        for group_keys, group_df in groups:
            # Normalize group_keys to handle different number of grouping columns
            if len(group_cols) == 1:
                subj, sess, run = group_keys, "session_0", "run_0"
            elif len(group_cols) == 2:
                subj, sess = group_keys
                run = "run_0"
            else:
                subj, sess, run = group_keys

            # These indices are relative to subset_metadata (e.g. metadata_main or metadata)
            abs_indices = group_df.index.values

            # Extract original metadata (excluding columns used for grouping)
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
        Executes advanced data splitting logic using MOABB evaluation strategies.

        This method partitions the input recordings into cross-validation folds (train/test sets)
        and handles optional validation data extraction. The process follows these stages:
        1. Validation & Initialization: Verifies configuration and instantiates the MOABB splitter.
        2. Data Aggregation: Collects signals, labels, and MOABB-required metadata (subject, session).
        3. Pre-split Validation: (Optional) Extracts a global validation set by withholding specific subjects.
        4. MOABB Partitioning: Delegates the core splitting logic to the instantiated MOABB strategy.
        5. Fold Packaging: Reconstructs RecordingDTOs for each fold and handles post-split validation.

        Args:
            input_dto (SplitInputDTO): Contains the MOABB evaluator configuration and input recordings.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[DatasetSplitDTO]: A result wrapping the generated folds (FoldDTOs) and optional
                global validation data.

        Raises:
            RuntimeError: If the splitter is disabled or if the MOABB partitioning logic fails.
            ValueError: If metadata aggregation fails or if data requirements are not met.
        """
        log = run_ctx.logger.for_step("MOABB_SPLITTER")
        config = input_dto.config
        recordings = input_dto.data.data
        dataset_name = recordings[0].dataset_name if recordings else "unknown"

        # Determine validation settings
        exclude_validation_data_before_split = config.exclude_validation_data_before_split
        validation_ratio = config.validation_ratio

        # --- 1. Validation & Initialization ---
        if not config.enabled:
            log.info("MoabbSplitter is disabled.", VerbosityLevel.QUIET)
            raise RuntimeError(f"MoabbSplitter is disabled. Check config settings for {type(self).__name__} to enable it or use a different splitter.")

        log.info("Starting MOABB Splitter", VerbosityLevel.QUIET)

        # Instantiate the MOABB splitter using Hydra
        evaluator_cfg = config.evaluator.model_dump(by_alias=True)

        if "cv_class" in evaluator_cfg and isinstance(evaluator_cfg["cv_class"], str):
            try:
                evaluator_cfg["cv_class"] = get_class(evaluator_cfg["cv_class"])
                log.info(f"Resolved cv_class to: {evaluator_cfg['cv_class']}", VerbosityLevel.TRACE)
            except Exception as e:
                log.error(f"Failed to resolve cv_class '{evaluator_cfg['cv_class']}': {e}")
                del evaluator_cfg["cv_class"]

        splitter = instantiate(evaluator_cfg)
        log.info(f"Successfully created splitter: {type(splitter).__name__}", VerbosityLevel.DETAILED)

        # --- 2. Data Aggregation ---
        x, y, metadata = self.extract_data(recordings)

        if metadata is None:
            log.error("Metadata aggregation failed or no recordings provided. MOABB splitter requires valid metadata (subject, session, run) for partitioning.")
            raise ValueError("Metadata aggregation failed or no recordings provided. MOABB splitter requires valid metadata (subject, session, run) for partitioning.")

        validation_data_global = None
        main_indices = np.arange(len(y))

        # --- 3. Pre-split Validation ---
        if validation_ratio > 0 and exclude_validation_data_before_split:
            random_state = config.evaluator.random_state
            if random_state is not None:
                np.random.seed(random_state)

            subjects = metadata["subject"].unique()
            num_val_subjects = max(1, int(len(subjects) * validation_ratio)) if len(subjects) > 1 else 0

            if num_val_subjects > 0:
                val_subjects = np.random.choice(subjects, num_val_subjects, replace=False)
                log.info(f"Global pre-split validation: extracted {num_val_subjects} subjects: {val_subjects}", VerbosityLevel.NORMAL)
                val_mask = metadata["subject"].isin(val_subjects)
                val_indices = np.where(val_mask)[0]
                main_indices = np.where(~val_mask)[0]

                if len(val_indices) > 0:
                    validation_data_global = self.create_dto(val_indices, x, y, metadata, dataset_name, "validation_global")
            else:
                log.error("Not enough subjects found. Consider using a different validation strategy or check subject metadata.")
                raise ValueError(f"Subject-based validation requested (ratio {validation_ratio}), but not enough subjects found (total subjects: {len(subjects)}). Consider using a different validation strategy or check subject metadata.")

        # Subset data for MOABB splitter
        x_main = x[main_indices]
        y_main = y[main_indices]
        metadata_main = metadata.iloc[main_indices].reset_index(drop=True)

        # --- Data Requirements Validation (After pre-split) ---
        main_subjects = metadata_main["subject"].unique()
        main_sessions = metadata_main["session"].unique()

        backend = getattr(config, "backend", "unknown")

        if backend == "moabb_cross_subject" and len(main_subjects) < 2:
            msg = (
                f"Cross-subject split requested, but only {len(main_subjects)} subject(s) available after validation extraction. "
                f"This splitter requires at least 2 subjects to perform cross-validation. "
                f"Please add more subjects, decrease validation_ratio, or use a different splitting strategy."
            )
            log.error(msg)
            raise ValueError(msg)

        if backend == "moabb_cross_session" and len(main_sessions) < 2:
            msg = (
                f"Cross-session split requested, but only {len(main_sessions)} session(s) available after validation extraction. "
                f"This splitter requires at least 2 sessions to perform cross-validation. "
                f"Please add more sessions or use a different splitting strategy."
            )
            log.error(msg)
            raise ValueError(msg)

        folds = []

        try:
            # --- 4. MOABB Partitioning ---
            log.info(f"Calling splitter.split with y shape {y_main.shape}", VerbosityLevel.TRACE)
            splits = list(splitter.split(y_main, metadata_main))
            log.info(f"Successfully generated {len(splits)} fold(s).", VerbosityLevel.TRACE)

            # --- 5. Fold Packaging ---
            all_val_indices = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                actual_train_idx = train_idx

                # If post-split validation is requested, we take a portion of the training indices for validation
                if validation_ratio > 0 and not exclude_validation_data_before_split:
                    num_val = int(len(train_idx) * validation_ratio)
                    if num_val > 0:
                        rs = config.evaluator.random_state
                        if rs is not None:
                            np.random.seed(rs + fold_idx)

                        shuffled_train_idx = np.random.permutation(train_idx)
                        val_idx_in_fold = shuffled_train_idx[:num_val]
                        actual_train_idx = shuffled_train_idx[num_val:]
                        all_val_indices.append(val_idx_in_fold)
                    log.info(f"Fold {fold_idx}: added {num_val} samples to validation pool.", VerbosityLevel.TRACE)

                log.info(f"Packaging fold {fold_idx}: extracting train/test DTOs", VerbosityLevel.NORMAL)
                train_dto = self.create_dto(actual_train_idx, x_main, y_main, metadata_main, dataset_name, "train")
                test_dto = self.create_dto(test_idx, x_main, y_main, metadata_main, dataset_name, "test")

                folds.append(FoldDTO(fold_idx=fold_idx, train_data=train_dto, test_data=test_dto))

            # If we gather validation indices from all folds, we can create a global validation set
            if all_val_indices:
                combined_val_idx = np.unique(np.concatenate(all_val_indices))
                validation_data_global = self.create_dto(combined_val_idx, x_main, y_main, metadata_main, dataset_name, "validation_global_post")

        except ValueError as e:
            # Re-raise descriptive ValueErrors (e.g. from MOABB about insufficient data)
            log.error(f"MOABB splitter raised ValueError: {e}. This likely indicates that the data does not meet the requirements for the requested split (e.g., not enough subjects/sessions). Check the error message for details.")
            raise RuntimeError(f"MOABB splitter {type(splitter).__name__} failed: {e}. Check if the data satisfies the splitter's requirements (e.g., enough subjects/sessions for the requested split).") from e
        except Exception as e:
            log.error(f"MOABB splitter {type(splitter).__name__} failed with an unexpected error: {e}")
            raise

        # --- Final Sanity Checks ---
        if not folds and len(folds) == 0:
            msg = f"Splitting process failed to generate any folds. Check your dataset and splitter configuration."
            log.error(msg)
            raise ValueError(msg)

        if validation_data_global is None or not validation_data_global.data:
            msg = f"No global validation data produced (validation_ratio: {validation_ratio}). The pipeline requires validation data for evaluation. Please ensure validation_ratio > 0 and that you have enough data/subjects."
            log.error(msg)
            raise ValueError(msg)

        result_data = DatasetSplitDTO(folds=folds, validation_data=validation_data_global)
        return StepResult(result_data)
