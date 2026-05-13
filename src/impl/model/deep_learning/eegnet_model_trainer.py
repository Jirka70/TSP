import logging

from src.impl.model.deep_learning.eegnet_model import EEGNetModel
from src.impl.model.util.extract.extract_learning_data import extract_learning_data
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.dto.split.dataset_split_dto import FoldDTO
from src.types.interfaces.model.model_trainer import IModelTrainer

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

        if not input_dto.folds:
            raise ValueError("EEGNet training needs at least one fold.")

        trained_models: list[TrainedModelDTO] = []

        for fold in input_dto.folds:
            trained_model = self.train_fold(fold=fold, validation_data=None, config=input_dto.config, run_ctx=run_ctx)

            trained_models.append(trained_model)

        return StepResult(TrainingResultDTO(trained_models))

    def train_fold(self, validation_data: EpochPreprocessedDTO | None, fold: FoldDTO, config: EEGNetConfig, run_ctx: RunContext):
        x_train, y_train = extract_learning_data(fold.train_data)
        fold_test_data = fold.test_data

        network = create_eegnet_network(config, x_train.shape)
        model = EEGNetModel(network=network, model_name=config.model_name, config=config)

        x_fold_test_data = None
        y_fold_test_data = None

        if fold_test_data is not None:
            x_fold_test_data, y_fold_test_data = extract_learning_data(fold_test_data)

        x_validation_data, y_validation_data = None, None

        if validation_data is not None:
            x_validation_data, y_validation_data = extract_learning_data(validation_data)

        model.initialize_training(y_train)
        best_validation_accuracy = None
        best_epoch = None

        for epoch in range(config.training.epochs):
            train_loss, train_accuracy = model.train_one_epoch(x_train, y_train)

            if x_fold_test_data is not None and y_fold_test_data is not None:
                fold_test_loss, fold_test_accuracy = model.validate(x_fold_test_data, y_fold_test_data)

            if x_validation_data is not None and y_validation_data is not None:
                validation_data_loss, validation_data_accuracy = model.evaluate(x_validation_data, y_validation_data)

                if best_validation_accuracy is None or validation_data_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_data_accuracy
                    best_epoch = epoch

        return TrainedModelDTO(
            model=model,
            model_name=config.model_name,
            history=model.history,
            best_epoch=best_epoch,
            best_validation_metric_name="accuracy" if best_validation_accuracy is not None else None,
            best_validation_metric_value=best_validation_accuracy,
            fold_idx=fold.fold_idx,
            metadata={
                "training_mode": "single_fold_training",
                "run_id": run_ctx.run_id,
                "n_train_samples": len(y_train),
                "n_fold_test_samples": len(y_fold_test_data) if y_fold_test_data is not None else 0,
                "n_validation_samples": len(y_validation_data) if y_validation_data is not None else 0,
            },
        )

