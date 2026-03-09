from src.types.dto.training.trained_model_dto import TrainedModelDTO
from src.types.dto.training.training_input_dto import TrainingInputDTO
from src.types.interfaces.model_trainer import IModelTrainer


class ModelTrainingStage:
    def __init__(self, trainer: IModelTrainer) -> None:
        self._trainer = trainer

    def run(self, input_dto: TrainingInputDTO) -> TrainedModelDTO:
        return self._trainer.run(input_dto)