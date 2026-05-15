from dataclasses import dataclass

from src.types.dto.load.recording import RecordingDTO


@dataclass(frozen=True)
class RawAugmentedDTO:
    """
    Data transfer object representing the signal after raw augmentation.
    """
    data: list[RecordingDTO]
