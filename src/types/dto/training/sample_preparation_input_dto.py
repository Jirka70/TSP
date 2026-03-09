from dataclasses import dataclass

from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO


@dataclass(frozen=True)
class SamplePreparationInputDTO:
    """
    Represents entry contract for sample preparation state
    """

    preprocessed_data: PreprocessedDataDTO
    """
    Data, which came from preprocessing pipeline stage
    """