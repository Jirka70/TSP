from src.pipeline.context.run_context import RunContext
from src.types.dto.load.data_loading_input_dto import DataLoadingInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class DummyLoader(IDataLoader):
    def run(self, input_dto: DataLoadingInputDTO, run_ctx: RunContext) -> RawDataDTO:
        data = RawDataDTO(sampling_freq=42, channel_names=[], signal="")
        return data