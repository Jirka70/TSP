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

        # 1. Export Validation Data
        if data.validation_data:
            self._export_epoch_dto(data.validation_data, export_dir / "validation", "validation")

        # 2. Export Folds
        for fold in data.folds:
            fold_name = f"fold_{fold.fold_idx}"
            if fold.train_data:
                self._export_epoch_dto(fold.train_data, export_dir / "fold_train", f"{fold_name}_train")
            if fold.test_data:
                self._export_epoch_dto(fold.test_data, export_dir / "fold_test", f"{fold_name}_test")

        # 3. Create Metadata
        self._create_metadata_json(config, data, export_dir)

        return StepResult(None)

    def _export_epoch_dto(self, dto: EpochPreprocessedDTO, target_dir: Path, prefix: str):
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, rec in enumerate(dto.data):
            # Reconstruct MNE Epochs
            sfreq = rec.metadata.get("sfreq", 250.0)  # Default fallback
            ch_names = rec.metadata.get("ch_names")

            if ch_names is None:
                # Fallback if names are missing - generate generic ones
                n_channels = rec.data.shape[1]
                ch_names = [f"EEG {j:03d}" for j in range(n_channels)]

            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

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
        metadata = {
            "backend": config.backend,
            "num_folds": len(data.folds),
            "has_validation": data.validation_data is not None,
            "folds": []
        }

        for fold in data.folds:
            metadata["folds"].append({
                "fold_idx": fold.fold_idx,
                "train_samples": sum(rec.data.shape[0] for rec in fold.train_data.data) if fold.train_data else 0,
                "test_samples": sum(rec.data.shape[0] for rec in fold.test_data.data) if fold.test_data else 0
            })

        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
