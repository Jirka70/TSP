"""MOABB splitting module for advanced data partitioning strategies."""

import logging

import mne
import numpy as np
import pandas as pd
from hydra.utils import get_class, instantiate

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.splitter import ISplitter

log = logging.getLogger(__name__)


class MoabbSplitter(ISplitter):
    """
    Splitter that uses MOABB (Mother of All BCI Benchmarks) evaluation strategies.

    This splitter interfaces with MOABB's evaluation splitters (WithinSession,
    CrossSession, CrossSubject, etc.) to generate cross-validation folds that
    respect the specific structure of EEG datasets.
    """

    def _extract_data(self, recordings: list) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
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
                log.warning(f"No labels found for recording {rec.subject_id}. Supervised learning may fail.")
                labels = np.zeros(len(signal))

            all_signals.append(signal)
            all_labels.append(labels)

            # 2. Metadata handling - MOABB requires 'subject' and 'session' columns
            if isinstance(rec.metadata, dict):
                # Ensure we have standardized subject and session identifiers
                subj = rec.metadata.get("subject", rec.subject_id)
                sess = rec.metadata.get("session", rec.session_id or "session_0")

                # Create a row template for this recording's epochs
                row_data = {"subject": subj, "session": sess}

                # Include other scalar metadata fields
                for k, v in rec.metadata.items():
                    if k not in ["subject", "session", "labels"] and np.isscalar(v):
                        row_data[k] = v

                # Replicate the row for every epoch in this recording
                df_meta = pd.DataFrame([row_data] * len(labels))
                all_metadata.append(df_meta)

        # Merge all recordings into unified structures
        combined_signal = np.concatenate(all_signals, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else None

        return combined_signal, combined_labels, combined_metadata

    def run(self, input_dto: SplitInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Executes the MOABB-based splitting logic.

        Args:
            input_dto: Contains the MOABB evaluator configuration and input data.
            run_ctx: Context of the current pipeline execution.

        Returns:
            StepResult containing a list of generated FoldDTOs.
        """
        config = input_dto.config
        recordings = input_dto.data.data

        # 1. Handle disabled splitter
        if not config.enabled:
            log.info("MoabbSplitter is disabled. Passing all data to training in Fold 0.")
            single_fold = FoldDTO(
                fold_idx=0,
                train_data=input_dto.data,
                validation_data=None,
                test_data=None,
            )
            return StepResult(DatasetSplitDTO(folds=[single_fold]))

        log.info("Starting MOABB Splitter")

        # 2. Instantiate the MOABB splitter using Hydra
        evaluator_cfg = config.evaluator.model_dump(by_alias=True)

        # Resolve the scikit-learn CV class if provided as a string path
        if "cv_class" in evaluator_cfg and isinstance(evaluator_cfg["cv_class"], str):
            try:
                evaluator_cfg["cv_class"] = get_class(evaluator_cfg["cv_class"])
                log.info(f"Resolved cv_class to: {evaluator_cfg['cv_class']}")
            except Exception as e:
                log.error(f"Failed to resolve cv_class '{evaluator_cfg['cv_class']}': {e}")
                # Removing the key to let MOABB use its default if resolution fails
                del evaluator_cfg["cv_class"]

        # Create the actual MOABB splitter instance
        splitter = instantiate(evaluator_cfg)
        log.info(f"Successfully created splitter: {type(splitter).__name__}")

        # 3. Aggregate data from all input recordings
        x, y, metadata = self._extract_data(recordings)

        # Ensure minimal metadata exists for MOABB requirements
        if metadata is None:
            log.warning("Metadata is missing! Creating dummy metadata with single subject/session.")
            metadata = pd.DataFrame({"subject": [1] * len(y), "session": ["session_0"] * len(y)})

        folds = []

        try:
            # 4. Generate splits using MOABB logic
            log.info(f"Calling splitter.split with y shape {y.shape} and metadata rows {len(metadata)}")
            splits = list(splitter.split(y, metadata))
            log.info(f"Successfully generated {len(splits)} fold(s) from {type(splitter).__name__}.")

            # 5. Package each split into a FoldDTO
            for fold_idx, (train_idx, test_idx) in enumerate(splits):

                def create_dto(idx_list: np.ndarray) -> EpochPreprocessedDTO:
                    """Internal helper to wrap a subset into the DTO structure."""
                    from src.types.dto.load.recording import RecordingDTO

                    # Package indices into a single 'combined' recording
                    subset_rec = RecordingDTO(data=x[idx_list], dataset_name=recordings[0].dataset_name if recordings else "unknown", subject_id="combined", session_id="combined", run_id="combined", metadata={"labels": y[idx_list]})
                    return EpochPreprocessedDTO(data=[subset_rec])

                train_dto = create_dto(train_idx)
                test_dto = create_dto(test_idx)

                folds.append(FoldDTO(fold_idx=fold_idx, train_data=train_dto, validation_data=None, test_data=test_dto))

        except Exception as e:
            # Fallback logic if MOABB splitting fails (e.g., due to data constraints)
            log.error(f"Error calling splitter.split(y, metadata): {e}")
            log.info("Falling back to a single 80/20 random split (Fold 0).")

            indices = np.random.permutation(len(y))
            split_point = int(0.8 * len(indices))
            train_idx, test_idx = indices[:split_point], indices[split_point:]

            def create_dto_fallback(idx_list: np.ndarray) -> EpochPreprocessedDTO:
                from src.types.dto.load.recording import RecordingDTO

                subset_rec = RecordingDTO(data=x[idx_list], dataset_name=recordings[0].dataset_name if recordings else "unknown", subject_id="combined", session_id="combined", run_id="combined", metadata={"labels": y[idx_list]})
                return EpochPreprocessedDTO(data=[subset_rec])

            folds.append(FoldDTO(fold_idx=0, train_data=create_dto_fallback(train_idx), validation_data=None, test_data=create_dto_fallback(test_idx)))

        result_data = DatasetSplitDTO(folds=folds)
        return StepResult(result_data)
