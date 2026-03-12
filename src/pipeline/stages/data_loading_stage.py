from src.pipeline.context.run_context import RunContext
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class DataLoadingStage:
    def __init__(self, loader: IDataLoader) -> None:
        self._loader = loader

    def run(self, input: DatasetConfig, run_ctx: RunContext) -> RawDataDTO:
        return self._loader.run(input, run_ctx)