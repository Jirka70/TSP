from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.types.dto.model.deep_learning.training_loop_result import TrainingLoopResult
from src.types.dto.model.learning_dataset import LearningDataset


class EEGNetTrainingLoop:

    @staticmethod
    def train(model: EEGNetModel,
              train_data: LearningDataset,
              validation_data:
              LearningDataset | None,
              epochs: int) -> TrainingLoopResult:
        model.initialize_training(train_data.y)

        best_epoch: int | None = None
        best_validation_accuracy: float | None = None
        best_state_dict: dict | None = None

        for epoch in range(epochs):
            model.train_one_epoch(train_data.x, train_data.y)

            if validation_data is None: continue

            _, validation_accuracy = model.validate(validation_data.x, validation_data.y)

            is_best_accuracy = best_validation_accuracy is None or validation_accuracy > best_validation_accuracy
            if is_best_accuracy:
                best_epoch = epoch
                best_validation_accuracy = validation_accuracy
                best_state_dict = model.get_network_state_dict()

        if best_state_dict is not None:
            model.load_network_state_dict(best_state_dict)
            model.best_epoch = best_epoch
            model.best_validation_accuracy = best_validation_accuracy
        else:
            model.best_epoch = epochs - 1 if epochs > 0 else None

        return TrainingLoopResult(
            best_epoch=model.best_epoch,
            best_validation_accuracy=model.best_validation_accuracy,
            best_state_dict=best_state_dict
        )

