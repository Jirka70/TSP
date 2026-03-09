from abc import ABC, abstractmethod

from src.types.dto.training.trained_model_dto import TrainedModelDTO
from src.types.dto.training.training_input_dto import TrainingInputDTO


class IModelTrainer(ABC):
    @abstractmethod
    def run(self, input_dto: TrainingInputDTO) -> TrainedModelDTO:
        raise NotImplementedError
