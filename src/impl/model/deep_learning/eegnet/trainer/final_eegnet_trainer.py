"""Train EEGNet on explicit train/val inputs and expose the selected model."""

from src.impl.model.deep_learning.eegnet.trainer.eegnet_training_loop import EEGNetTrainingLoop
from src.impl.model.deep_learning.factory.deep_learning_model_factory import DeepLearningModelFactory
from src.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed
from src.impl.model.deep_learning.validation.validate_dl_input import validate_dl_input
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

STEP_NAME = "EEGNET FINAL TRAINING"


class FinalEEGNetTrainer(IFinalTrainer):
    def __init__(self, model_factory: DeepLearningModelFactory | None = None) -> None:
        if model_factory is None:
            model_factory = DeepLearningModelFactory()
        self._model_factory = model_factory

    def run(
        self,
        input_dto: FinalTrainingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[FinalTrainingResultDTO]:
        validated_input = validate_dl_input(input_dto)
        configuration = input_dto.config
        if not isinstance(configuration, EEGNetConfig):
            raise ValueError("FinalEEGNetTrainer requires EEGNetConfig.")

        logger = run_ctx.logger.for_step(STEP_NAME)
        logger.info("Started training EEGNET model", VerbosityLevel.QUIET)
        logger.info(f"Class counts: {validated_input.class_counts}", VerbosityLevel.DETAILED)
        for warning in validated_input.warnings:
            logger.warning(warning)

        training_dataset = validated_input.training
        validation_dataset = validated_input.validation
        logger.info(
            f"Training samples={training_dataset.sample_count}, validation samples={validation_dataset.sample_count}",
            VerbosityLevel.DETAILED,
        )

        seed = configuration.training.random_state
        if seed is not None:
            set_torch_seed(seed, configuration.training.deterministic)
            logger.info(f"Applied training seed={seed}", VerbosityLevel.TRACE)

        input_shape = training_dataset.x.shape
        model = self._model_factory.create(
            config=configuration,
            input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        )
        training_result = EEGNetTrainingLoop.train(
            model=model,
            train_data=training_dataset,
            validation_data=validation_dataset,
            epochs=configuration.training.epochs,
            log=logger,
        )

        best_validation_metric_name: str | None = None
        if training_result.best_validation_accuracy is not None:
            best_validation_metric_name = "accuracy"

        trained_model = TrainedModelDTO(
            model=model,
            model_name=configuration.model_name,
            history=model.history,
            best_epoch=training_result.best_epoch,
            best_validation_metric_name=best_validation_metric_name,
            best_validation_metric_value=training_result.best_validation_accuracy,
            metadata={
                "training_mode": "final_training",
                "run_id": run_ctx.run_id,
                "n_train_samples": training_dataset.sample_count,
                "n_validation_samples": validation_dataset.sample_count,
                "training_data_source": "training_data",
            },
            verification_data=validation_dataset,
        )
        logger.info("Finished training EEGNET model", VerbosityLevel.QUIET)
        return StepResult(FinalTrainingResultDTO(trained_model=trained_model))
