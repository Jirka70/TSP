from dataclasses import dataclass

from src.types.dto.config.astageconfig import AStageConfig
from src.types.dto.load.raw_data_dto import RawDataDTO


@dataclass(frozen=True)
class PreprocessingInputDTO:
    raw_data: RawDataDTO
    preprocessingConfig: AStageConfig
