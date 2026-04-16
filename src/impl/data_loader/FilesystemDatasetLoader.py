from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.interfaces.data_loader import IDataLoader
from mne.io import read_raw


class FilesystemDatasetLoader(IDataLoader):
    def run(self, input: FilesystemDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        path: str = input.path
        recordings: list[RecordingDTO] = []

        for subject_directory in self._file_discovery.iter_directories(root_path):
            subject_id = self._subject_id_parser.parse(subject_directory.name)
            if subject_id is None:
                continue

            if allowed_subject_ids is not None and subject_id not in allowed_subject_ids:
                continue

            recordings.extend(
                self._load_subject_directory(
                    subject_directory=subject_directory,
                    dataset_name=dataset_name,
                    subject_id=subject_id,
                    recursive=config.recursive,
                )
            )

        pass

