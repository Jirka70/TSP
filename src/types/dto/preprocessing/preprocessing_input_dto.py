from dataclasses import dataclass

from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.config.astageconfig import AStageConfig


@dataclass(frozen=True)
class PreprocessingInputDTO:
    raw_data: RawDataDTO
    preprocessingConfig: AStageConfig
