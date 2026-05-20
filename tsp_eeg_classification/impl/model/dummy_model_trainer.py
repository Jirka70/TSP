import logging

from tsp_eeg_classification.impl.model.dummy_model import DummyModel
from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.model.trained_model_dto import TrainedModelDTO
from tsp_eeg_classification.types.dto.model.training_input_dto import TrainingInputDTO
from tsp_eeg_classification.types.interfaces.model.model import IModel
from tsp_eeg_classification.types.interfaces.model.model_trainer import IModelTrainer


class DummyModelTrainer(IModelTrainer):
    def run(
        self, input_dto: TrainingInputDTO, run_ctx: RunContext
    ) -> StepResult[TrainedModelDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy model trainer")
        model: IModel = DummyModel()
        data: TrainedModelDTO = TrainedModelDTO(model, model_name="eegnet")
        return StepResult(data)
