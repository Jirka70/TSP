from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.load.data_loading_input_dto import DataLoadingInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO


class IDataLoader(ABC):
    @abstractmethod
    def run(self, input_dto: DataLoadingInputDTO, run_ctx: RunContext) -> RawDataDTO:
        raise NotImplementedError
