from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentedSamplesDTO:
    X: object
    y: object
    metadata: dict[str, object]