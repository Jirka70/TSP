from abc import ABC, abstractmethod

from src.types.dto.load.data_loading_input_dto import DataLoadingInputDTO
from src.types.dto.raw_data_dto import RawDataDTO


class IDataLoader(ABC):
    @abstractmethod
    def run(self, input_dto: DataLoadingInputDTO) -> RawDataDTO:
        raise NotImplementedError