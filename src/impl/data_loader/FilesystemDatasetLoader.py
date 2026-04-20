import logging
from pathlib import Path
from typing import Any

import mne
import pandas as pd
from mne.io import BaseRaw

from src.impl.data_loader.util.file_discovery import FileDiscovery
from src.impl.data_loader.util.mne_raw_reader import MneRawReader
from src.impl.data_loader.util.subject_id_parser import SubjectIdParser
from src.impl.data_loader.util.supported_recording_file_policy import SupportedRecordingFilePolicy
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.interfaces.data_loader import IDataLoader


class FilesystemDatasetLoader(IDataLoader):


    def __init__(self):
        self._log = logging.getLogger(__name__)

    def run(self, input: FilesystemDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        root_path: Path = Path(input.path)
        global_annotations: mne.Annotations | None = self._load_global_annotations(input)
        recordings: list[RecordingDTO] = []
        allowed_subject_ids = set(input.subject_ids) if input.subject_ids else None

        for subject_directory in FileDiscovery.iter_directories(root_path):
            subject_id = SubjectIdParser.parse(subject_directory.name)
            if subject_id is None:
                self._log.warning(f"Subject id from folder {subject_directory} cannot be obtained. Skipping this folder...")
                continue

            if allowed_subject_ids is not None and subject_id not in allowed_subject_ids:
                self._log.warning(f"Skipping folder {subject_directory.name} with subject_id {subject_id}")
                continue

            new_recordings: list[RecordingDTO] = self._load_subject_directory(
                subject_directory=subject_directory,
                subject_id=subject_id,
                dataset_name="dataset",
                recursive=input.recursive,
                global_annotations=global_annotations,
            )

            recordings.extend(new_recordings)

        res: RawDataDTO = RawDataDTO(recordings)
        return StepResult(res)

    def _load_subject_directory(
        self,
        *,
        subject_directory: Path,
        dataset_name: str | None,
        subject_id: int,
        recursive: bool,
        global_annotations: mne.Annotations | None,
    ) -> list[RecordingDTO]:
        subject_recordings: list[RecordingDTO] = []

        for file_path in FileDiscovery.iter_files(subject_directory, recursive):
            if not SupportedRecordingFilePolicy.is_supported(file_path):
                continue

            raw_data: BaseRaw = MneRawReader.read(file_path)
            has_annotations: bool = raw_data.annotations is not None and len(raw_data.annotations) > 0
            annotation_source: str = "embedded"

            if not has_annotations and global_annotations is not None:
                fitted_annotations = self._fit_annotations_to_recording(raw_data, global_annotations)
                if fitted_annotations is not None:
                    raw_data.set_annotations(fitted_annotations)
                    annotation_source = "global_tsv"
                else:
                    annotation_source = "none"
                    self._log.warning("No global events annotations fit recording duration for %s. Skipping annotation enrichment.", file_path.name)
            elif not has_annotations:
                annotation_source = "none"

            new_recording: RecordingDTO = self._create_recording_dto(
                raw_data=raw_data,
                file_path=file_path,
                dataset_name=dataset_name,
                subject_id=subject_id,
                annotation_source=annotation_source,
            )

            subject_recordings.append(new_recording)

        return subject_recordings

    def _create_recording_dto(
        self,
        *,
        raw_data: Any,
        file_path: Path,
        dataset_name: str,
        subject_id: int,
        annotation_source: str,
    ) -> RecordingDTO:
        return RecordingDTO(
            data=raw_data,
            dataset_name=dataset_name,
            subject_id=subject_id,
            session_id=None,
            run_id=None,
            metadata={
                "source_file": str(file_path),
                "file_name": file_path.name,
                "directory": str(file_path.parent),
                "annotation_source": annotation_source,
            },
        )

    def _load_global_annotations(self, config: FilesystemDatasetConfig) -> mne.Annotations | None:
        if config.global_events_tsv_path is None:
            return None

        events_path = Path(config.global_events_tsv_path)
        if not events_path.exists():
            self._log.warning("Global events TSV does not exist: %s", events_path)
            return None

        events_df = pd.read_csv(events_path, sep="\t")
        required_columns = {"onset", "duration"}
        if not required_columns.issubset(events_df.columns):
            self._log.warning("Global events TSV at %s is missing required columns: %s", events_path, required_columns)
            return None

        onset = pd.to_numeric(events_df["onset"], errors="coerce")
        duration = pd.to_numeric(events_df["duration"], errors="coerce")
        if onset.isna().any() or duration.isna().any():
            self._log.warning("Global events TSV at %s contains non-numeric onset/duration values.", events_path)
            return None

        # Dataset stores event timing in milliseconds, MNE expects seconds.
        onset_seconds = onset.astype(float) / 1000.0
        duration_seconds = duration.astype(float) / 1000.0

        description = events_df["trial_type"].astype(str) if "trial_type" in events_df.columns else pd.Series(["event"] * len(events_df))

        return mne.Annotations(
            onset=onset_seconds.to_numpy(),
            duration=duration_seconds.to_numpy(),
            description=description.to_numpy(),
        )

    def _fit_annotations_to_recording(self, raw_data: BaseRaw, annotations: mne.Annotations) -> mne.Annotations | None:
        recording_duration_seconds = raw_data.times[-1]
        annotation_start = annotations.onset
        annotation_end = annotations.onset + annotations.duration

        # Keep only rows that overlap with the actual recording and clip overflows.
        overlaps_recording = (annotation_end > 0.0) & (annotation_start < recording_duration_seconds)
        if not overlaps_recording.any():
            return None

        clipped_onset = annotation_start[overlaps_recording].copy()
        clipped_duration = annotations.duration[overlaps_recording].copy()
        clipped_description = annotations.description[overlaps_recording].copy()

        negative_start_mask = clipped_onset < 0.0
        clipped_duration[negative_start_mask] = clipped_duration[negative_start_mask] + clipped_onset[negative_start_mask]
        clipped_onset[negative_start_mask] = 0.0

        overflow_mask = clipped_onset + clipped_duration > recording_duration_seconds
        clipped_duration[overflow_mask] = recording_duration_seconds - clipped_onset[overflow_mask]

        positive_duration_mask = clipped_duration > 0.0
        if not positive_duration_mask.any():
            return None

        return mne.Annotations(
            onset=clipped_onset[positive_duration_mask],
            duration=clipped_duration[positive_duration_mask],
            description=clipped_description[positive_duration_mask],
        )
