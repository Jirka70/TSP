import logging

from pipeline.context.run_context import RunContext
from pipeline.contracts.step_result import StepResult
from types.dto.model.training_input_dto import TrainingInputDTO
from types.dto.model.training_result_dto import TrainingResultDTO
from types.interfaces.model.model_trainer import IModelTrainer

log = logging.getLogger(__name__)


class EEGNetModelTrainer(IModelTrainer):
    def run(
        self,
        input_dto: TrainingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[TrainingResultDTO]:
        log.info("Starting EEGNet fold training. Run: %s", run_ctx.run_id)
        log.info("Number of folds: %s", len(input_dto.folds))

        if not input_dto.config.fold_training:
            log.info("Fold training is disabled. Skipping fold training stage.")
            return StepResult(TrainingResultDTO(trained_models=[]))