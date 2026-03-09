from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    backend: str | None