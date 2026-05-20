from dataclasses import dataclass

from tsp_eeg_classification.types.dto.load.recording import RecordingDTO


@dataclass(frozen=True)
class RawAugmentedDTO:
    """
    Data transfer object representing the signal after raw augmentation.
    """
    data: list[RecordingDTO]
