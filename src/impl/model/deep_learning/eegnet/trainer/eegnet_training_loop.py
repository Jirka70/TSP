"""Train epochs and restore the network selected by validation accuracy."""

from torch import Tensor

from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.model.deep_learning.training_loop_result import TrainingLoopResult
from src.types.dto.model.learning_dataset import LearningDataset


class EEGNetTrainingLoop:
    @staticmethod
    def train(
        model: EEGNetModel,
        train_data: LearningDataset,
        validation_data: LearningDataset | None,
        log: PipelineLogger,
        epochs: int,
    ) -> TrainingLoopResult:
        """Select the earliest best val epoch; without val keep the last epoch.

        Epoch indices in the returned metadata start at zero. The main DL trainer
        requires val through its input validator; optional val remains supported
        here for compatibility with existing direct callers.
        """
        log.info("Initializing EEGNET training...")
        model.initialize_training(train_data.y)

        best_epoch: int | None = None
        best_validation_accuracy: float | None = None
        best_network_state: dict[str, Tensor] | None = None

        for epoch_index in range(epochs):
            training_loss, training_accuracy = model.train_one_epoch(train_data.x, train_data.y)
            log.info(
                f"Epoch {epoch_index + 1} train loss={training_loss}, accuracy={training_accuracy}",
                VerbosityLevel.DETAILED,
            )
            if validation_data is None:
                continue

            validation_loss, validation_accuracy = model.validate(validation_data.x, validation_data.y)
            log.info(
                f"Epoch {epoch_index + 1} validation loss={validation_loss}, accuracy={validation_accuracy}",
                VerbosityLevel.DETAILED,
            )
            is_better_validation_accuracy = (
                best_validation_accuracy is None or validation_accuracy > best_validation_accuracy
            )
            if is_better_validation_accuracy:
                best_epoch = epoch_index
                best_validation_accuracy = validation_accuracy
                best_network_state = model.get_network_state_dict()

        if best_network_state is not None:
            model.load_network_state_dict(best_network_state)
        elif epochs > 0:
            best_epoch = epochs - 1

        model.best_epoch = best_epoch
        model.best_validation_accuracy = best_validation_accuracy
        return TrainingLoopResult(
            best_epoch=best_epoch,
            best_validation_accuracy=best_validation_accuracy,
            best_state_dict=best_network_state,
        )
