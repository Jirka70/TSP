import logging

import numpy as np

from src.impl.model.eegnet_model import EEGNetModel
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
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

        trained_models: list[TrainedModelDTO] = []

        for fold in input_dto.folds:
            log.info("Training EEGNet on fold %s", fold.fold_idx)

            x_train, y_train = self._extract_data_and_labels(fold.train_data)

            x_val = None
            y_val = None
            if fold.validation_data is not None:
                x_val, y_val = self._extract_data_and_labels(fold.validation_data)

            network = create_eegnet_network(input_dto.config, shape=x_train.shape)

            # get n_channels from any fold
            # network.n_chans = fold.train_data.data[0].metadata["n_channels"]

            model = EEGNetModel(
                network=network,
                model_name="eegnet",
                config=input_dto.config,
            )

            history = model.fit_with_validation(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )

            trained_models.append(
                TrainedModelDTO(
                    model=model,
                    model_name="eegnet",
                    history=history,
                    best_epoch=model.best_epoch,
                    best_validation_metric_name="accuracy" if model.best_validation_accuracy is not None else None,
                    best_validation_metric_value=model.best_validation_accuracy,
                    fold_idx=fold.fold_idx,
                    metadata={
                        "training_mode": "fold_training",
                        "run_id": run_ctx.run_id,
                        "n_train_samples": len(y_train),
                        "n_validation_samples": len(y_val) if y_val is not None else 0,
                    },
                )
            )

        return StepResult(
            TrainingResultDTO(
                trained_models=trained_models,
            )
        )

    def _extract_data_and_labels(
        self,
        data_dto: EpochPreprocessedDTO,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_list = []
        y_list = []

        for recording in data_dto.data:
            epochs = recording.data

            if hasattr(epochs, "get_data"):
                x_list.append(epochs.get_data(copy=False))
                y_list.append(epochs.events[:, -1])
            else:
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        if x.ndim != 3:
            raise ValueError(f"EEGNet expects input shape (n_epochs, n_channels, n_times), got {x.shape}")

        return x, y