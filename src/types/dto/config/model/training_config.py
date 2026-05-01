from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Hyperparameters used by model training backends."""

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
