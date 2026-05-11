import logging

import numpy as np

from src.impl.model.eegnet_model import EEGNetModel
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

log = logging.getLogger(__name__)


class FinalEEGNetTrainer(IFinalTrainer):
    """
    Trains one final EEGNet model.

    This trainer is not for fold-based evaluation. Prefer passing train_data that
    contains each final-training sample once. Cross-validation folds are accepted
    only as a backward-compatible fallback.
    """

    def run(
        self,
        input_dto: FinalTrainingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[FinalTrainingResultDTO]:
        network = create_eegnet_network(input_dto.config)
        model = EEGNetModel(network=network,
                            model_name=input_dto.config.model_name,
                            config=input_dto.config)

        all_y_train = np.concatenate([
            self._extract_data_and_labels(fold.train_data)[1]
            for fold in input_dto.folds
        ])

        return None
        """log.info("Starting final EEGNet training. Run: %s", run_ctx.run_id)

        x_val = None
        y_val = None
        if input_dto.train_data is not None:
            x_train, y_train = self._extract_data_and_labels(input_dto.train_data)
            training_source = "full_dataset"
        else:
            if not input_dto.folds:
                raise ValueError("Final EEGNet training needs train_data or at least one fallback fold.")

            fold = input_dto.folds[0]
            x_train, y_train = self._extract_data_and_labels(fold.train_data)
            training_source = "first_fold_train_data"

            if fold.validation_data is not None:
                x_val, y_val = self._extract_data_and_labels(fold.validation_data)

        network = create_eegnet_network(input_dto.config)

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

        trained_model = TrainedModelDTO(
            model=model,
            model_name="eegnet",
            history=history,
            best_epoch=model.best_epoch,
            best_validation_metric_name="accuracy"
            if model.best_validation_accuracy is not None
            else None,
            best_validation_metric_value=model.best_validation_accuracy,
            fold_idx=None,
            metadata={
                "training_mode": "final_training",
                "training_source": training_source,
                "run_id": run_ctx.run_id,
                "n_train_samples": len(y_train),
                "n_validation_samples": len(y_val) if y_val is not None else 0,
                "test_data_used_for_training": False,
            },
        )

        log.info(
            "Final EEGNet model trained. train_samples=%s validation_samples=%s best_epoch=%s best_validation_accuracy=%s",
            len(y_train),
            len(y_val) if y_val is not None else 0,
            model.best_epoch,
            model.best_validation_accuracy,
        )

        return StepResult(
            FinalTrainingResultDTO(
                trained_model=trained_model,
            )
        )"""

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
            raise ValueError(
                f"EEGNet expects input shape "
                f"(n_epochs, n_channels, n_times), got {x.shape}"
            )

        return x, y
