# src/types/dto/epoching/epoching_data_dto.py

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpochingDataDTO:
    data: Any
    """
    Backendový objekt s epochami.
    Typicky mne.Epochs nebo jiná reprezentace.
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