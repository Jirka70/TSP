from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt, field_validator


class TrainingConfig(BaseModel):
    """Hyperparameters used by model training backends."""

    model_config = ConfigDict(extra="forbid")

    epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat = Field(le=1.0)
    optimizer: Literal["adam", "sgd"]

    # Reproducibility of training
    random_state: NonNegativeInt | None = 67
    deterministic: bool = False

    @field_validator("optimizer", mode="before")
    @classmethod
    def normalize_optimizer(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.lower()
