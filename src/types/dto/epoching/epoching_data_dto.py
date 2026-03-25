from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpochingDataDTO:
    data: Any
    """
    mne.Epochs or different representation.
    """

    labels: list[int]
    """
    Label for each epoch.
    """

    event_names: list[str]
    """
    Event name for each epoch
    """

    sampling_rate_hz: float
    n_epochs: int
    n_channels: int
    n_times: int

    channel_names: list[str]
    metadata: dict[str, Any] | None = None
