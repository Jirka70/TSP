from src.impl.model.deep_learning.eegnet.trainer.eegnet_training_loop import EEGNetTrainingLoop
from src.impl.model.deep_learning.factory.deep_learning_model_factory import DeepLearningModelFactory
from src.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed
from src.impl.model.util.extract.extract_learning_data import extract_learning_data
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.learning_dataset import LearningDataset
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

STEP_NAME = "EEGNET FINAL TRAINING"


def _extract_final_training_data(
        input_dto: FinalTrainingInputDTO,
) -> LearningDataset:
    if input_dto.training_data is not None:
        return extract_learning_data(input_dto.training_data)

    raise ValueError(
        "Final EEGNet training received no train_data. "
        "Final EEGNet training requires training_data."
    )


class FinalEEGNetTrainer(IFinalTrainer):

    def __init__(self, model_factory: DeepLearningModelFactory | None = None) -> None:
        self._model_factory = model_factory or DeepLearningModelFactory()

    def run(
            self,
            input_dto: FinalTrainingInputDTO,
            run_ctx: RunContext,
    ) -> StepResult[FinalTrainingResultDTO]:
        log = run_ctx.logger.for_step(STEP_NAME)
        log.info("Started training EEGNET model", VerbosityLevel.QUIET)

        log.info("Extracting training data", VerbosityLevel.DETAILED)
        train_data = _extract_final_training_data(input_dto)
        log.info(f"Training data extracted: samples={train_data.sample_count} shape={train_data.x.shape}",
                 VerbosityLevel.TRACE)

        seed = input_dto.config.training.random_state
        if seed is not None:
            set_torch_seed(seed, input_dto.config.training.deterministic)
            log.info("Seed from input config was applied successfully", VerbosityLevel.TRACE)
        else:
            log.info("Seed was set to None. No initial seed is being applied...", VerbosityLevel.TRACE)

        model = self._model_factory.create(config=input_dto.config, input_shape=train_data.x.shape)
        log.info("EEGNET model created successfully")

        validation_data = None
        if input_dto.validation_data is not None:
            log.info("Extracting validation data", VerbosityLevel.TRACE)
            validation_data = extract_learning_data(input_dto.validation_data)
            log.info(f"Validation data extracted: samples={validation_data.sample_count}"
                     f"shape={validation_data.x.shape}", VerbosityLevel.TRACE)
        else:
            log.warning("Validation data are not present. Model will not be validated")

        training_result = EEGNetTrainingLoop.train(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            epochs=input_dto.config.training.epochs,
            log=log
        )

        best_epoch = training_result.best_epoch
        best_validation_metric_name = "accuracy" if training_result.best_validation_accuracy is not None else None
        training_data_source = "train_data" if input_dto.training_data is not None else "fold_fallback"

        model.best_epoch = training_result.best_epoch
        model.best_validation_accuracy = training_result.best_validation_accuracy

        trained_model = TrainedModelDTO(
            model=model,
            model_name=input_dto.config.model_name,
            history=model.history,
            best_epoch=best_epoch,
            fold_idx=None,
            best_validation_metric_name=best_validation_metric_name,
            best_validation_metric_value=training_result.best_validation_accuracy,
            metadata={
                "training_mode": "final_training",
                "run_id": run_ctx.run_id,
                "n_folds": len(input_dto.folds),
                "n_train_samples": train_data.sample_count,
                "n_validation_samples": validation_data.sample_count if validation_data is not None else 0,
                "training_data_source": training_data_source
            },
        )
        log.info("Finished training EEGNET model", VerbosityLevel.QUIET)
        return StepResult(
            FinalTrainingResultDTO(
                trained_model=trained_model,
            )
        )
