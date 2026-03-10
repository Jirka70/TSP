from src.pipeline.context.run_context import RunContext
from src.types.dto.load.data_loading_input_dto import DataLoadingInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class DataLoadingStage:
    def __init__(self, loader: IDataLoader) -> None:
        self._loader = loader

    def run(self, input_dto: DataLoadingInputDTO, run_ctx: RunContext) -> RawDataDTO:
        return self._loader.run(input_dto, run_ctx)