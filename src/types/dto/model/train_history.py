from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingHistory:
    """History of training"""

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)

    train_metrics: dict[str, list[float]] = field(default_factory=dict)
    validation_metrics: dict[str, list[float]] = field(default_factory=dict)
