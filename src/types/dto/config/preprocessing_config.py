from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    backend: str
    l_freq: float | None
    h_freq: float | None
    notch_freq: float | None
    sampling_rate_hz: float | None
    rereference: str | None
    channel_selection: list[str] | None