from dataclasses import dataclass

from src.types.dto.load.raw_data_dto import RawDataDTO


@dataclass(frozen=True)
class PreprocessingInputDTO:
    raw_data: RawDataDTO
    l_freq: float | None
    h_freq: float | None
    notch_freq: float | None
    target_s_freq: float | None
    channel_selection: list[str] | None