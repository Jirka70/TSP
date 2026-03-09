from abc import ABC, abstractmethod

from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO


class IPreprocessing(ABC):
    @abstractmethod
    def run(self, input_dto: PreprocessingInputDTO) -> PreprocessedDataDTO:
        raise NotImplementedError
