from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawDataDTO:
    signal: Any
    """
    Loaded EEG signal
    """

    sampling_freq: float
    """
    Sampling frequency of signal. How often the signal was sampled per one second
    """

    channel_names: list[str]
    """
    List of EEG channel names
    """

    metadata: dict[str, object] | None = None
    """
    Optional container for information about loaded data (dataset name, run id, etc.)
    """