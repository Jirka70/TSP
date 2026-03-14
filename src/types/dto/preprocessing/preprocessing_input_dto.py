from dataclasses import dataclass

from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.config.iconfig import IConfig


@dataclass(frozen=True)
class PreprocessingInputDTO:
    raw_data: RawDataDTO
    preprocessingConfig: IConfig
