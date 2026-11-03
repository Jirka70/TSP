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

    Each fold contains train_data and test_data only — no global validation data
    is extracted. The fold's test set serves as the held-out evaluation partition.
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
        Executes data splitting using MOABB evaluation strategies.

        Partitions the input recordings into cross-validation folds (train/test sets).
        No global validation data is extracted — each fold's test set is the held-out
        evaluation partition. The process follows these stages:
        1. Initialization: Instantiates the MOABB splitter from config.
        2. Data Aggregation: Collects signals, labels, and MOABB-required metadata.
        3. Data Requirements Validation: Checks subject/session counts for the strategy.
        4. MOABB Partitioning: Delegates the core splitting logic to the MOABB strategy.
        5. Fold Packaging: Reconstructs RecordingDTOs for each fold.

        Args:
            input_dto (SplitInputDTO): Contains the MOABB evaluator configuration and input recordings.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[DatasetSplitDTO]: A result wrapping the generated folds (FoldDTOs).
                validation_data is always None — use fold test sets for evaluation.

        Raises:
            RuntimeError: If the MOABB partitioning logic fails.
            ValueError: If metadata aggregation fails or data requirements are not met.
        """
        log = run_ctx.logger.for_step("MOABB_SPLITTER")
        config = input_dto.config
        recordings = input_dto.data.data
        dataset_name = recordings[0].dataset_name if recordings else "unknown"

        log.info("Starting MOABB Splitter", VerbosityLevel.QUIET)

        # --- 1. Initialization ---
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

        # --- 3. Data Requirements Validation ---
        subjects = metadata["subject"].unique()
        sessions = metadata["session"].unique()
        backend = getattr(config, "backend", "unknown")

        if backend == "moabb_cross_subject" and len(subjects) < 2:
            msg = f"Cross-subject split requested, but only {len(subjects)} subject(s) available. This splitter requires at least 2 subjects to perform cross-validation. Please add more subjects or use a different splitting strategy."
            log.error(msg)
            raise ValueError(msg)

        if backend == "moabb_cross_session" and len(sessions) < 2:
            msg = f"Cross-session split requested, but only {len(sessions)} session(s) available. This splitter requires at least 2 sessions to perform cross-validation. Please add more sessions or use a different splitting strategy."
            log.error(msg)
            raise ValueError(msg)

        # --- 4. MOABB Partitioning ---
        folds = []

        try:
            log.info(f"Calling splitter.split with y shape {y.shape}", VerbosityLevel.TRACE)
            splits = list(splitter.split(y, metadata))
            log.info(f"Successfully generated {len(splits)} fold(s).", VerbosityLevel.TRACE)

            # --- 5. Fold Packaging ---
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                log.info(f"Packaging fold {fold_idx}: extracting train/test DTOs", VerbosityLevel.NORMAL)
                train_dto = self.create_dto(train_idx, x, y, metadata, dataset_name, "train")
                test_dto = self.create_dto(test_idx, x, y, metadata, dataset_name, "test")
                folds.append(FoldDTO(fold_idx=fold_idx, train_data=train_dto, test_data=test_dto))

        except ValueError as e:
            # Re-raise descriptive ValueErrors (e.g. from MOABB about insufficient data)
            log.error(f"MOABB splitter raised ValueError: {e}. This likely indicates that the data does not meet the requirements for the requested split (e.g., not enough subjects/sessions). Check the error message for details.")
            raise RuntimeError(f"MOABB splitter {type(splitter).__name__} failed: {e}. Check if the data satisfies the splitter's requirements (e.g., enough subjects/sessions for the requested split).") from e
        except Exception as e:
            log.error(f"MOABB splitter {type(splitter).__name__} failed with an unexpected error: {e}")
            raise

        # --- Final Sanity Check ---
        if not folds:
            msg = "Splitting process failed to generate any folds. Check your dataset and splitter configuration."
            log.error(msg)
            raise ValueError(msg)

        return StepResult(DatasetSplitDTO(folds=folds, validation_data=None))
