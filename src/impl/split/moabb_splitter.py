"""MOABB splitting module for advanced data partitioning strategies."""

import logging

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

log = logging.getLogger(__name__)


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
        combined_signal = np.concatenate(all_signals, axis=0) if all_signals else np.array([])
        combined_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        combined_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else None

        return combined_signal, combined_labels, combined_metadata

    def run(self, input_dto: SplitInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        """
        Executes the MOABB-based splitting logic.

        Args:
            input_dto: Contains the MOABB evaluator configuration and input data.
            run_ctx: Context of the current pipeline execution.

        Returns:
            StepResult containing a list of generated FoldDTOs and global validation data.
        """
        config = input_dto.config
        recordings = input_dto.data.data

        # Determine validation settings
        pre_split_val = config.pre_split_validation
        val_ratio = config.validation_ratio

        # 1. Handle disabled splitter
        if not config.enabled:
            log.info("MoabbSplitter is disabled. Passing all data to training in Fold 0.")
            single_fold = FoldDTO(
                fold_idx=0,
                train_data=input_dto.data,
                test_data=None,
            )
            return StepResult(DatasetSplitDTO(folds=[single_fold], validation_data=None))

        log.info("Starting MOABB Splitter")

        # 2. Instantiate the MOABB splitter using Hydra
        evaluator_cfg = config.evaluator.model_dump(by_alias=True)

        if "cv_class" in evaluator_cfg and isinstance(evaluator_cfg["cv_class"], str):
            try:
                evaluator_cfg["cv_class"] = get_class(evaluator_cfg["cv_class"])
                log.info(f"Resolved cv_class to: {evaluator_cfg['cv_class']}")
            except Exception as e:
                log.error(f"Failed to resolve cv_class '{evaluator_cfg['cv_class']}': {e}")
                del evaluator_cfg["cv_class"]

        splitter = instantiate(evaluator_cfg)
        log.info(f"Successfully created splitter: {type(splitter).__name__}")

        # 3. Aggregate data from all input recordings
        x, y, metadata = self.extract_data(recordings)

        if metadata is None:
            log.warning("Metadata is missing! Creating dummy metadata with single subject/session.")
            metadata = pd.DataFrame({"subject": [1] * len(y), "session": ["session_0"] * len(y)})

        validation_data_global = None
        main_indices = np.arange(len(y))

        # 4. Handle pre-split validation
        if val_ratio > 0 and pre_split_val:
            random_state = config.evaluator.random_state
            if random_state is not None:
                np.random.seed(random_state)

            subjects = metadata["subject"].unique()
            num_val_subjects = max(1, int(len(subjects) * val_ratio)) if len(subjects) > 1 else 0

            if num_val_subjects > 0:
                val_subjects = np.random.choice(subjects, num_val_subjects, replace=False)
                log.info(f"Global pre-split validation: extracted {num_val_subjects} subjects: {val_subjects}")
                val_mask = metadata["subject"].isin(val_subjects)
                val_indices = np.where(val_mask)[0]
                main_indices = np.where(~val_mask)[0]

                if len(val_indices) > 0:
                    val_rec = RecordingDTO(
                        data=x[val_indices],
                        dataset_name=recordings[0].dataset_name if recordings else "unknown",
                        subject_id="validation_global",
                        # todo session_id by subject -> možná to bude potřeba někde dále, tak proč z toho dělat combined
                        session_id="combined",
                        run_id="combined",
                        metadata={"labels": y[val_indices]},
                    )
                    validation_data_global = EpochPreprocessedDTO(data=[val_rec])
            else:
                # todo spis hodit exception -> nedostatek subjektů pro validaci
                log.warning("Subject-based validation requested but not enough subjects found.")

        # Subset data for MOABB splitter
        x_main = x[main_indices]
        y_main = y[main_indices]
        metadata_main = metadata.iloc[main_indices].reset_index(drop=True)

        folds = []

        def create_dto(idx_list: np.ndarray, subset_x: np.ndarray, subset_y: np.ndarray, name_prefix: str = "combined") -> EpochPreprocessedDTO | None:
            if idx_list is None or len(idx_list) == 0:
                return None
            subset_rec = RecordingDTO(
                data=subset_x[idx_list],
                dataset_name=recordings[0].dataset_name if recordings else "unknown",
                subject_id=name_prefix,
                session_id="combined",
                run_id="combined",
                metadata={"labels": subset_y[idx_list]},
            )
            return EpochPreprocessedDTO(data=[subset_rec])

        try:
            # 5. Generate splits using MOABB logic
            log.info(f"Calling splitter.split with y shape {y_main.shape}")
            splits = list(splitter.split(y_main, metadata_main))
            log.info(f"Successfully generated {len(splits)} fold(s).")

            # 6. Package each split into a FoldDTO and handle post-split validation
            all_val_indices = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                actual_train_idx = train_idx

                # If post-split validation is requested, we take a portion of the training indices for validation
                if val_ratio > 0 and not pre_split_val:
                    num_val = int(len(train_idx) * val_ratio)
                    if num_val > 0:
                        rs = config.evaluator.random_state
                        if rs is not None:
                            np.random.seed(rs + fold_idx)

                        shuffled_train_idx = np.random.permutation(train_idx)
                        val_idx_in_fold = shuffled_train_idx[:num_val]
                        actual_train_idx = shuffled_train_idx[num_val:]
                        all_val_indices.append(val_idx_in_fold)
                    log.info(f"Global validation data was added {len(all_val_indices)} subjects.")
                train_dto = create_dto(actual_train_idx, x_main, y_main, "train_combined")
                test_dto = create_dto(test_idx, x_main, y_main, "test_combined")

                folds.append(FoldDTO(fold_idx=fold_idx, train_data=train_dto, test_data=test_dto))

            # If we gather validation indices from all folds, we can create a global validation set
            if all_val_indices:
                combined_val_idx = np.concatenate(all_val_indices)
                combined_val_idx = np.unique(combined_val_idx)
                val_dto = create_dto(combined_val_idx, x_main, y_main, "validation_global_post")
                if val_dto:
                    validation_data_global = val_dto

        except Exception as e:
            # todo tady prostě hodit exception -> doporučit jiný splitter / jiné nastavení splitu
            log.error(f"Error calling splitter.split: {e}")
            log.info("Falling back to a single random split (Fold 0).")
            indices = np.random.permutation(len(y_main))
            test_ratio = 0.2
            num_test = int(len(indices) * test_ratio)
            test_idx = indices[:num_test]
            train_val_idx = indices[num_test:]

            actual_train_idx = train_val_idx
            if val_ratio > 0 and not pre_split_val:
                num_val = int(len(train_val_idx) * val_ratio)
                if num_val > 0:
                    val_idx = train_val_idx[:num_val]
                    actual_train_idx = train_val_idx[num_val:]
                    val_dto = create_dto(val_idx, x_main, y_main, "val_fallback")
                    if val_dto:
                        validation_data_global = val_dto

            train_dto = create_dto(actual_train_idx, x_main, y_main, "train_combined")
            test_dto = create_dto(test_idx, x_main, y_main, "test_combined")
            folds.append(FoldDTO(fold_idx=0, train_data=train_dto, test_data=test_dto))

        result_data = DatasetSplitDTO(folds=folds, validation_data=validation_data_global)
        return StepResult(result_data)
