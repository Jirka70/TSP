from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    backend: str
    copies_per_sample: int
    gaussian_noise_std: float
    max_time_shift: int
    channel_dropout_prob: float
