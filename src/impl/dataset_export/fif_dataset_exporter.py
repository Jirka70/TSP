import json
import logging
from pathlib import Path

import mne
import numpy as np

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_export_config import DatasetExportConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.dataset_exporter import IDatasetExporter

log = logging.getLogger(__name__)


class FifDatasetExporter(IDatasetExporter):
    """
    Exports the augmented dataset to MNE .fif files.

    Supports two modes driven by the contents of :class:`DatasetSplitDTO`:

    - **Training pipeline** (``folds=None``): exports ``train``, ``validation``,
      and ``test`` partitions to separate sub-directories.
    - **Evaluation pipeline** (``folds`` is set): exports each fold's
      ``train_data`` and ``test_data`` to ``fold_train`` / ``fold_test``
      sub-directories.
    """

    def run(
        self, config: DatasetExportConfig, data: DatasetSplitDTO, run_ctx: RunContext
    ) -> StepResult[None]:
        if not config.enabled:
            log.info("Dataset export is disabled.")
            return StepResult(None)

        export_dir = run_ctx.output_dir / "dataset_export"
        export_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Exporting augmented dataset to {export_dir}")

        if data.folds is None:
            # --- Training pipeline mode: export train / validation / test directly ---
            if data.train:
                self._export_epoch_dto(data.train, export_dir / "train", "train")
            if data.validation:
                self._export_epoch_dto(data.validation, export_dir / "validation", "validation")
            if data.test:
                self._export_epoch_dto(data.test, export_dir / "test", "test")
        else:
            # --- Evaluation pipeline mode: export per-fold data ---
            for fold in data.folds:
                fold_name = f"fold_{fold.fold_idx}"
                if fold.train_data:
                    self._export_epoch_dto(fold.train_data, export_dir / "fold_train", f"{fold_name}_train")
                if fold.test_data:
                    self._export_epoch_dto(fold.test_data, export_dir / "fold_test", f"{fold_name}_test")

        # Create Metadata JSON
        self._create_metadata_json(config, data, export_dir)

        return StepResult(None)

    def _export_epoch_dto(self, dto: EpochPreprocessedDTO, target_dir: Path, prefix: str):
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, rec in enumerate(dto.data):
            # Reconstruct MNE Epochs
            sfreq = rec.metadata.get("sfreq", 250.0)  # Default fallback
            channel_names = rec.metadata.get("channel_names") or rec.metadata.get("ch_names")

            if channel_names is None:
                # Fallback if names are missing - generate generic ones
                n_channels = rec.data.shape[1]
                channel_names = [f"EEG {j:03d}" for j in range(n_channels)]

            info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")

            # Create dummy events
            n_epochs = rec.data.shape[0]
            events = np.zeros((n_epochs, 3), dtype=int)
            events[:, 0] = np.arange(n_epochs) * int(sfreq)  # Fake onsets
            if "labels" in rec.metadata:
                events[:, 2] = rec.metadata["labels"]

            epochs = mne.EpochsArray(rec.data, info, events=events, verbose=False)

            # Generate filename
            filename = f"{prefix}_subj_{rec.subject_id}"
            if rec.session_id:
                filename += f"_sess_{rec.session_id}"
            if rec.run_id:
                filename += f"_run_{rec.run_id}"
            filename += "-epo.fif"

            epochs.save(target_dir / filename, overwrite=True, verbose=False)

    def _create_metadata_json(self, config: DatasetExportConfig, data: DatasetSplitDTO, export_dir: Path):
        if data.folds is None:
            # Training pipeline mode
            metadata = {
                "backend": config.backend,
                "mode": "training",
                "has_train": data.train is not None,
                "has_validation": data.validation is not None,
                "has_test": data.test is not None,
                "train_samples": sum(rec.data.shape[0] for rec in data.train.data) if data.train else 0,
                "validation_samples": sum(rec.data.shape[0] for rec in data.validation.data) if data.validation else 0,
                "test_samples": sum(rec.data.shape[0] for rec in data.test.data) if data.test else 0,
            }
        else:
            # Evaluation pipeline mode
            metadata = {
                "backend": config.backend,
                "mode": "evaluation",
                "num_folds": len(data.folds),
                "folds": [],
            }
            for fold in data.folds:
                metadata["folds"].append({
                    "fold_idx": fold.fold_idx,
                    "train_samples": sum(rec.data.shape[0] for rec in fold.train_data.data) if fold.train_data else 0,
                    "test_samples": sum(rec.data.shape[0] for rec in fold.test_data.data) if fold.test_data else 0,
                })

        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
