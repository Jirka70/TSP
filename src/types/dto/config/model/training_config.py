from dataclasses import dataclass

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
