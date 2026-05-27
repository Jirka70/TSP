from src.impl.model.deep_learning.eegnet.trainer.eegnet_training_loop import EEGNetTrainingLoop
from src.impl.model.deep_learning.factory.deep_learning_model_factory import DeepLearningModelFactory
from src.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed
from src.impl.model.util.extract.extract_learning_data import extract_learning_data
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.dto.split.dataset_split_dto import FoldDTO
from src.types.interfaces.model.model_trainer import IModelTrainer


class EEGNetModelTrainer(IModelTrainer):
    def __init__(self, model_factory: DeepLearningModelFactory | None = None) -> None:
        self._model_factory = model_factory or DeepLearningModelFactory()

    def run(
            self,
            input_dto: TrainingInputDTO,
            run_ctx: RunContext,
    ) -> StepResult[TrainingResultDTO]:
        log = run_ctx.logger.for_step("")
        log.info(f"Starting EEGNet fold training. Run: {run_ctx.run_id}", VerbosityLevel.QUIET)
        log.info(f"Number of folds: {len(input_dto.folds)}")

        if not input_dto.config.fold_training:
            log.info("Fold training is disabled. Skipping fold training stage.")
            return StepResult(TrainingResultDTO(trained_models=[]))

        if not input_dto.folds:
            raise ValueError("EEGNet training needs at least one fold.")

        trained_models: list[TrainedModelDTO] = []

        seed = input_dto.config.training.random_state

        for fold in input_dto.folds:
            if seed is not None:
                set_torch_seed(seed + fold.fold_idx, input_dto.config.training.deterministic)
            trained_model = self.train_fold(fold=fold, config=input_dto.config, run_ctx=run_ctx)

            trained_models.append(trained_model)

        return StepResult(TrainingResultDTO(trained_models=trained_models))

    def train_fold(self, fold: FoldDTO, config: EEGNetConfig, run_ctx: RunContext) -> TrainedModelDTO:
        train_data = extract_learning_data(fold.train_data)

        model = self._model_factory.create(config=config, input_shape=train_data.x.shape)

        validation_data = (
            extract_learning_data(fold.test_data)
            if fold.test_data is not None
            else None
        )

        training_result = EEGNetTrainingLoop.train(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            epochs=config.training.epochs,
        )

        return TrainedModelDTO(
            model=model,
            model_name=config.model_name,
            history=model.history,
            best_epoch=training_result.best_epoch,
            best_validation_metric_name="accuracy" if training_result.best_validation_accuracy is not None else None,
            best_validation_metric_value=training_result.best_validation_accuracy,
            fold_idx=fold.fold_idx,
            metadata={
                "training_mode": "single_fold_training",
                "run_id": run_ctx.run_id,
                "n_train_samples": train_data.sample_count,
                "n_fold_test_samples": validation_data.sample_count if validation_data is not None else 0,
            },
        )
