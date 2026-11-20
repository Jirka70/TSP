import json
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.dataset_export_config import DatasetExportConfig
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.dataset_exporter import IDatasetExporter


class FifDatasetExporter(IDatasetExporter):
    """
    Exports the (augmented) dataset splits to MNE ``-epo.fif`` files.

    The export layout mirrors the split produced by the selected pipeline:

    - **Training pipeline** (``folds is None``)::

        dataset_export/
            metadata.json
            train/        # final-model training data (augmented)
            validation/   # only when present (DL models, validation_ratio > 0)
            test/         # only when present (test_ratio > 0)

    - **Benchmark pipeline** (``folds`` is set)::

        dataset_export/
            metadata.json
            fold_0/
                train/    # data for training the fold's (small) ML model
                test/     # held-out data for evaluating that fold model
            fold_1/
                ...

    Each split directory contains one file per recording (subject / session / run).
    Signals that are not epoched time series (e.g. CSP features shaped
    ``(n_epochs, n_components)``) cannot be stored as ``-epo.fif`` and are exported
    as ``.npz`` (``X``, ``y``) instead.
    """

    def run(self, config: DatasetExportConfig, data: DatasetSplitDTO, run_ctx: RunContext) -> StepResult[None]:
        log = run_ctx.logger.for_step("DATASET_EXPORT")

        export_dir = run_ctx.output_dir / "dataset_export"
        export_dir.mkdir(parents=True, exist_ok=True)

        mode = self._resolve_mode(data, log)
        log.info(f"Exporting dataset ({mode} pipeline) to {export_dir}", VerbosityLevel.QUIET)

        metadata: dict[str, Any] = {
            "backend": config.backend,
            "pipeline": mode,
            "pipeline_name": run_ctx.pipeline_name,
            "run_id": run_ctx.run_id,
            "experiment_name": run_ctx.experiment_name,
        }

        if mode == "training":
            splits = {
                "train": data.train,
                "validation": data.validation,
                "test": data.test,
            }
            metadata["has_validation"] = data.validation is not None
            metadata["has_test"] = data.test is not None
            metadata["splits"] = {}
            for split_name, split_dto in splits.items():
                if split_dto is None:
                    log.info(f"Split '{split_name}' is empty, skipping.", VerbosityLevel.DETAILED)
                    continue
                metadata["splits"][split_name] = self._export_split(split_dto, export_dir / split_name, export_dir, log)
        else:
            metadata["num_folds"] = len(data.folds)
            metadata["folds"] = []
            # --- Benchmark pipeline mode: export per-fold data ---
            for fold in data.folds:
                fold_dir = export_dir / f"fold_{fold.fold_idx}"
                fold_meta: dict[str, Any] = {"fold_idx": fold.fold_idx, "splits": {}}
                log.info(f"Exporting fold {fold.fold_idx}", VerbosityLevel.DETAILED)
                for split_name, split_dto in (("train", fold.train_data), ("test", fold.test_data)):
                    if split_dto is None:
                        log.warning(f"Fold {fold.fold_idx} has no '{split_name}' data, skipping.")
                        continue
                    fold_meta["splits"][split_name] = self._export_split(split_dto, fold_dir / split_name, export_dir, log)
                metadata["folds"].append(fold_meta)

        with open(export_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(metadata), f, indent=4)

        log.info("Dataset export completed.", VerbosityLevel.QUIET)
        return StepResult(None)

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_mode(data: DatasetSplitDTO, log: PipelineLogger) -> str:
        """Determines the pipeline mode from the split DTO and validates its consistency."""
        if data.folds is not None:
            if not data.folds:
                msg = "DatasetSplitDTO contains an empty fold list. Nothing to export for the evaluation pipeline."
                log.error(msg)
                raise ValueError(msg)
            if any(part is not None for part in (data.train, data.validation, data.test)):
                log.warning(
                    "DatasetSplitDTO contains both folds and train/validation/test partitions. "
                    "Exporting folds only (evaluation pipeline)."
                )
            return "evaluation"

        if data.train is None:
            msg = "DatasetSplitDTO contains neither folds nor train data. Nothing to export."
            log.error(msg)
            raise ValueError(msg)
        return "training"

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_split(self, dto: EpochPreprocessedDTO, target_dir: Path, export_root: Path, log: PipelineLogger) -> dict[str, Any]:
        """Exports all recordings of a single split and returns its summary."""
        target_dir.mkdir(parents=True, exist_ok=True)

        files: list[str] = []
        label_counts: Counter = Counter()
        subjects: set[str] = set()
        n_samples = 0
        used_names: set[str] = set()

        for rec in dto.data:
            signal, labels, event_id, epochs = self._extract_signal_and_labels(rec)
            n_samples += len(labels)
            label_counts.update(str(lbl) for lbl in labels)
            subjects.add(str(rec.subject_id))

            stem = self._unique_stem(self._build_stem(rec), used_names)

            if epochs is not None or signal.ndim == 3:
                if epochs is None:
                    epochs = self._build_epochs(rec, signal, labels, event_id)
                path = target_dir / f"{stem}-epo.fif"
                epochs.save(path, overwrite=True, verbose=False)
            else:
                log.warning(
                    f"Recording '{stem}' has signal of shape {signal.shape} (not epochs x channels x times); "
                    "exporting as .npz instead of .fif."
                )
                path = target_dir / f"{stem}.npz"
                np.savez(path, X=signal, y=labels)

            files.append(path.relative_to(export_root).as_posix())

        log.info(
            f"Exported {n_samples} samples from {len(files)} recording(s) to {target_dir}",
            VerbosityLevel.NORMAL,
        )

        return {
            "n_samples": n_samples,
            "n_recordings": len(files),
            "subjects": sorted(subjects),
            "label_counts": dict(sorted(label_counts.items())),
            "files": files,
        }

    @staticmethod
    def _extract_signal_and_labels(rec: RecordingDTO) -> tuple[np.ndarray | None, np.ndarray, dict[str, int] | None, mne.BaseEpochs | None]:
        """
        Returns ``(signal, labels, event_id, epochs)``.

        If the recording already holds MNE Epochs (e.g. splitter disabled), they are
        returned as ``epochs`` and saved as-is.
        """
        if isinstance(rec.data, mne.BaseEpochs):
            epochs = rec.data
            return None, epochs.events[:, -1], dict(epochs.event_id), epochs

        signal = np.asarray(rec.data)
        if "labels" not in rec.metadata:
            raise ValueError(f"Recording of subject {rec.subject_id} has no 'labels' in metadata. Cannot export dataset.")

        labels = np.asarray(rec.metadata["labels"])
        if len(labels) != signal.shape[0]:
            raise ValueError(
                f"Recording of subject {rec.subject_id}: number of labels ({len(labels)}) "
                f"does not match number of epochs ({signal.shape[0]})."
            )
        return signal, labels, rec.metadata.get("event_id"), None

    @staticmethod
    def _build_epochs(rec: RecordingDTO, signal: np.ndarray, labels: np.ndarray, event_id: dict[str, int] | None) -> mne.EpochsArray:
        """Reconstructs MNE Epochs from a numpy signal and its metadata."""
        n_epochs, n_channels, _ = signal.shape
        sfreq = float(rec.metadata.get("sfreq", 250.0))
        tmin = float(rec.metadata.get("tmin", 0.0))

        channel_names = rec.metadata.get("channel_names")
        if channel_names is None:
            channel_names = rec.metadata.get("ch_names")
        if channel_names is None or len(channel_names) != n_channels:
            channel_names = [f"EEG {j:03d}" for j in range(n_channels)]
        channel_names = [str(ch) for ch in channel_names]

        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")

        # Event codes must be integers; map non-integer labels (e.g. class names) to codes.
        if np.issubdtype(labels.dtype, np.integer):
            codes = labels.astype(int)
            if not event_id:
                event_id = {str(code): int(code) for code in np.unique(codes)}
        else:
            classes = sorted({str(lbl) for lbl in labels})
            event_id = {cls: idx + 1 for idx, cls in enumerate(classes)}
            codes = np.array([event_id[str(lbl)] for lbl in labels], dtype=int)

        # Keep only event_id entries actually present (MNE requires all codes to exist in events).
        present = set(np.unique(codes).tolist())
        event_id = {name: int(code) for name, code in event_id.items() if int(code) in present}

        events = np.zeros((n_epochs, 3), dtype=int)
        events[:, 0] = np.arange(n_epochs) * max(int(sfreq), 1)  # synthetic, strictly increasing onsets
        events[:, 2] = codes

        return mne.EpochsArray(signal, info, events=events, tmin=tmin, event_id=event_id, verbose=False)

    @staticmethod
    def _build_stem(rec: RecordingDTO) -> str:
        stem = f"subj_{rec.subject_id}"
        if rec.session_id:
            stem += f"_sess_{rec.session_id}"
        if rec.run_id is not None and rec.run_id != "":
            stem += f"_run_{rec.run_id}"
        return stem.replace("/", "_").replace("\\", "_").replace(" ", "_")

    @staticmethod
    def _unique_stem(stem: str, used: set[str]) -> str:
        candidate, i = stem, 1
        while candidate in used:
            candidate = f"{stem}_{i}"
            i += 1
        used.add(candidate)
        return candidate

    def _to_jsonable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
