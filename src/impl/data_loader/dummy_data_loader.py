from pipeline.context.run_context import RunContext
from types.interfaces.data_loader import IDataLoader
from types.dto.load.data_loading_input_dto import DataLoadingInputDTO
from types.dto.load.raw_data_dto import RawDataDTO

class DummyLoader(IDataLoader):
    def run(self, input_dto: DataLoadingInputDTO, run_ctx: RunContext) -> RawDataDTO:
        data = RawDataDTO()
        data.sampling_freq = 42
        return data