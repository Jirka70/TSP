from collections.abc import Sized

from src.impl.model.deep_learning.eegnet.trainer.eegnet_training_loop import EEGNetTrainingLoop
from src.impl.model.deep_learning.factory.deep_learning_model_factory import DeepLearningModelFactory
from src.impl.model.deep_learning.reproducibility.set_torch_seed import set_torch_seed
from src.impl.model.deep_learning.validation.validate_dl_input import validate_dl_input
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.dl_input_dto import DLInputDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

STEP_NAME = "EEGNET FINAL TRAINING"


def _has_recordings(dataset: EpochPreprocessedDTO | None) -> bool:
    return dataset is not None and bool(dataset.data)


def _require_dataset_with_epochs(dataset: EpochPreprocessedDTO | None, name: str) -> EpochPreprocessedDTO:
    if not _has_recordings(dataset):
        raise ValueError(f"Final EEGNet training requires a non-empty '{name}' dataset.")
    assert dataset is not None
    for recording in dataset.data:
        if _has_signal_epochs(recording.data):
            return dataset
    raise ValueError(f"Final EEGNet '{name}' dataset contains no epochs.")


def _has_signal_epochs(signal: object) -> bool:
    if not isinstance(signal, Sized):
        return False
    try:
        return len(signal) > 0
    except TypeError:
        return False


class FinalEEGNetTrainer(IFinalTrainer):
    def __init__(self, model_factory: DeepLearningModelFactory | None = None) -> None:
        self._model_factory = model_factory or DeepLearningModelFactory()

    def _to_dl_input(self, input_dto: FinalTrainingInputDTO) -> DLInputDTO:
        """Unpack one fold and a separate validation set without splitting data.

        Intended for BasicSplitter output. This checks presence, not dataset
        independence; full input validation is a separate responsibility.
        """
        if not isinstance(input_dto.config, EEGNetConfig):
            raise ValueError("FinalEEGNetTrainer requires EEGNetConfig.")

        if len(input_dto.folds) != 1:
            raise ValueError(
                f"Final EEGNet input conversion requires exactly one fold; received {len(input_dto.folds)}."
            )

        (fold,) = input_dto.folds
        return DLInputDTO(
            config=input_dto.config,
            train=_require_dataset_with_epochs(fold.train_data, "train"),
            val=_require_dataset_with_epochs(input_dto.validation_data, "val"),
            test=_require_dataset_with_epochs(fold.test_data, "test"),
        )

    def run(
        self,
        input_dto: FinalTrainingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[FinalTrainingResultDTO]:
        dl_input = self._to_dl_input(input_dto)
        validated_input = validate_dl_input(dl_input)
        log = run_ctx.logger.for_step(STEP_NAME)
        log.info("Started training EEGNET model", VerbosityLevel.QUIET)
        log.info(f"Class counts: {validated_input.class_counts}", VerbosityLevel.DETAILED)
        for warning in validated_input.warnings:
            log.warning(warning)

        log.info("Extracting training data", VerbosityLevel.DETAILED)
        train_data = validated_input.training
        log.info(
            f"Training data extracted: samples={train_data.sample_count} shape={train_data.x.shape}",
            VerbosityLevel.TRACE,
        )

        seed = dl_input.config.training.random_state
        if seed is not None:
            set_torch_seed(seed, dl_input.config.training.deterministic)
            log.info("Seed from input config was applied successfully", VerbosityLevel.TRACE)
        else:
            log.info("Seed was set to None. No initial seed is being applied...", VerbosityLevel.TRACE)

        model = self._model_factory.create(config=dl_input.config, input_shape=train_data.x.shape)
        log.info("EEGNET model created successfully")

        log.info("Extracting validation data", VerbosityLevel.TRACE)
        validation_data = validated_input.validation
        log.info(
            f"Validation data extracted: samples={validation_data.sample_count} shape={validation_data.x.shape}",
            VerbosityLevel.TRACE,
        )

        training_result = EEGNetTrainingLoop.train(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            epochs=dl_input.config.training.epochs,
            log=log,
        )

        best_epoch = training_result.best_epoch
        best_validation_metric_name = "accuracy" if training_result.best_validation_accuracy is not None else None
        training_data_source = "fold.train_data"

        model.best_epoch = training_result.best_epoch
        model.best_validation_accuracy = training_result.best_validation_accuracy

        trained_model = TrainedModelDTO(
            model=model,
            model_name=dl_input.config.model_name,
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
                "n_validation_samples": validation_data.sample_count,
                "training_data_source": training_data_source,
            },
        )
        log.info("Finished training EEGNET model", VerbosityLevel.QUIET)
        return StepResult(
            FinalTrainingResultDTO(
                trained_model=trained_model,
            )
        )
