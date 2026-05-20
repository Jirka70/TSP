from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Hyperparameters used by model training backends."""

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str

    # Reproducibility of training
    random_state: int | None = 67
    deterministic: bool = False
