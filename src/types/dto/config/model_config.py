from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    n_classes: int