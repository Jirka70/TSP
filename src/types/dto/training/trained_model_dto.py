from dataclasses import dataclass


@dataclass(frozen=True)
class TrainedModelDTO:
    model: object
    history: dict[str, object]
    metadata: dict[str, object]