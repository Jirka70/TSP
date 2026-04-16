from pathlib import Path
from typing import Any

from src.impl.data_loader.util.file_discovery import FileDiscovery
from src.impl.data_loader.util.mne_raw_reader import MneRawReader
from src.impl.data_loader.util.subject_id_parser import SubjectIdParser
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.interfaces.data_loader import IDataLoader


class FilesystemDatasetLoader(IDataLoader):
    def run(self, input: FilesystemDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        root_path: Path = Path(input.path)
        recordings: list[RecordingDTO] = []
        allowed_subject_ids = set(input.subject_ids) if input.subject_ids else None

        for subject_directory in FileDiscovery.iter_directories(root_path):
            subject_id = SubjectIdParser.parse(subject_directory.name)
            if subject_id is None:
                continue

            if allowed_subject_ids is not None and subject_id not in allowed_subject_ids:
                continue

            recordings.extend(
                self._load_subject_directory(
                    subject_directory=subject_directory,
                    subject_id=subject_id,
                    dataset_name="dataset",
                    recursive=input.recursive,
                )
            )

        res: RawDataDTO = RawDataDTO(recordings)
        return StepResult(res)

    def _load_subject_directory(
        self,
        *,
        subject_directory: Path,
        dataset_name: str | None,
        subject_id: int,
        recursive: bool,
    ) -> list[RecordingDTO]:
        subject_recordings: list[RecordingDTO] = []

        for file_path in FileDiscovery.iter_files(subject_directory, recursive):
            raw_data = MneRawReader.read(file_path)

            subject_recordings.append(
                self._create_recording_dto(
                    raw_data=raw_data,
                    file_path=file_path,
                    dataset_name=dataset_name,
                    subject_id=subject_id,
                )
            )

        return subject_recordings

    def _create_recording_dto(
        self,
        *,
        raw_data: Any,
        file_path: Path,
        dataset_name: str,
        subject_id: int,
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
            },
        )
