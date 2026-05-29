from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
from src.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class SyntheticDataLoader(IDataLoader):
    def run(self, input: ExternalDatasetConfig | FilesystemDatasetConfig, run_ctx: RunContext) -> StepResult[
        RawDataDTO]:
        pass