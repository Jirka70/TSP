import logging

from src.impl.model.dummy_model import DummyModel
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.interfaces.model.model import IModel
from src.types.interfaces.model.model_trainer import IModelTrainer


class DummyModelTrainer(IModelTrainer):
    def run(self, input_dto: TrainingInputDTO, run_ctx: RunContext) -> StepResult[TrainedModelDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy model trainer")
        model: IModel = DummyModel()
        data: TrainedModelDTO = TrainedModelDTO(model, model_name="pepa zetek")
        return StepResult(data)