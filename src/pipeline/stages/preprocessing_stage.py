from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.interfaces.preprocessing import IPreprocessing


class PreprocessingStage:
    def __init__(self, preprocessing_impl: IPreprocessing) -> None:
        self._preprocessing_impl = preprocessing_impl

    def run(self, input_dto: PreprocessingInputDTO) -> PreprocessedDataDTO:
        return self._preprocessing_impl.run(input_dto)
