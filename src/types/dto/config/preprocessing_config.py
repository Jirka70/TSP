from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    backend: str
    l_freq: float | None
    h_freq: float | None
    notch_freq: float | None
    target_sfreq: float | None
    rereference: str | None
    channel_selection: list[str] | None