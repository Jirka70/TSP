from dataclasses import dataclass

from src.types.dto.load.recording import RecordingDTO


@dataclass(frozen=True)
class RawDataDTO:
    data: list[RecordingDTO]
    """
    Loaded EEG signal with metadata
    """
