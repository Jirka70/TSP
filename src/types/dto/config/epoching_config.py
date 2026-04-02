from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class EpochingConfig(AStageConfig):
    backend: Literal["mne"]
    """
    Configuration of epoching step.
    """

    enabled: bool

    event_source: Literal["annotations", "stim_channel"]
    """
    Where event declarations come from
    """

    event_mapping: dict[str, int]
    """
    Mapping of event names to event ID in data
    E.g. {"left_hand": 2, "right_hand": 3}
    """

    event_labels: dict[str, int]
    """
    Mapping of event name on classification label (for training-training purposes)
    E.g. {"left_hand": 0, "right_hand": 1}
    """

    tmin: float
    tmax: float
    """
    Start and end time of epoch relative to event
    """

    baseline: tuple[float, float] | None
    """
    Např. (-0.2, 0.0) nebo None.
    """

    preload: bool
    reject_by_annotation: bool
    drop_last_incomplete_epoch: bool
    skip_missing_events: bool

    picks: list[str] | None
    """
    Optional channel pick
    """
