from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader
from mne.io import read_raw


class EDFLoader(IDataLoader):
    def run(self, input: FilesystemDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        return read_raw(input.pa)

