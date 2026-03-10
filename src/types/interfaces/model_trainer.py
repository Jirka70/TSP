from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.training.trained_model_dto import TrainedModelDTO
from src.types.dto.training.training_input_dto import TrainingInputDTO


from abc import ABC, abstractmethod


class IModelTrainer(ABC):

    @abstractmethod
    def fit(self, X, y, run_context: RunContext):
        pass

    @abstractmethod
    def predict(self, X, run_context: RunContext):
        pass
