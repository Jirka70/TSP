from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.model.deep_learning.training_loop_result import TrainingLoopResult
from src.types.dto.model.learning_dataset import LearningDataset


def _log_epoch_metrics(
        model: EEGNetModel,
        log: PipelineLogger,
        epoch: int,
) -> None:
    if model.history is None:
        log.warning(f"Epoch {epoch + 1} has no training history to summarize.")
        return

    train_accuracy = model.history.train_metrics.get("accuracy", [])
    if train_accuracy:
        log.info(f"Epoch {epoch + 1} train accuracy={train_accuracy[-1]}", VerbosityLevel.DETAILED)


class EEGNetTrainingLoop:

    @staticmethod
    def train(
            model: EEGNetModel,
            train_data: LearningDataset,
            validation_data: LearningDataset | None,
            log: PipelineLogger,
            epochs: int,
    ) -> TrainingLoopResult:
        log.info("Initializing EEGNET training...")
        model.initialize_training(train_data.y)

        best_epoch: int | None = None
        best_validation_accuracy: float | None = None
        best_state_dict: dict | None = None

        for epoch in range(epochs):
            model.train_one_epoch(train_data.x, train_data.y)

            if validation_data is None:
                continue

            _, validation_accuracy = model.validate(validation_data.x, validation_data.y)

            is_best_accuracy = best_validation_accuracy is None or validation_accuracy > best_validation_accuracy
            if is_best_accuracy:
                best_epoch = epoch
                best_validation_accuracy = validation_accuracy
                best_state_dict = model.get_network_state_dict()

            _log_epoch_metrics(model, log, epoch)

        if best_state_dict is not None:
            model.load_network_state_dict(best_state_dict)
            model.best_epoch = best_epoch
            model.best_validation_accuracy = best_validation_accuracy
        else:
            model.best_epoch = epochs - 1 if epochs > 0 else None
            model.best_validation_accuracy = None

        return TrainingLoopResult(
            best_epoch=model.best_epoch,
            best_validation_accuracy=model.best_validation_accuracy,
            best_state_dict=best_state_dict
        )
