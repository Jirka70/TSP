from typing import Any

from src.pipeline.context.run_context import RunContext
from src.types.interfaces.model_trainer import IModelTrainer


class ModelTrainingStage:
    def __init__(self, trainer: IModelTrainer) -> None:
        self._trainer = trainer

    def fit(self, X, y, run_ctx: RunContext) -> None:
        raise NotImplementedError

    def predict(self, X, run_ctx: RunContext) -> Any:
        raise NotImplementedError
