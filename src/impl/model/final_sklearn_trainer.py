import logging

import numpy as np
from sklearn.metrics import accuracy_score

from src.impl.model.generic_sklearn_model import GenericSklearnModel
from src.impl.model.model_factory import ModelFactory
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.train_history import TrainingHistory
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

log = logging.getLogger(__name__)


class FinalSklearnTrainer(IFinalTrainer):
    def run(self, input_dto: FinalTrainingInputDTO, run_ctx: RunContext) -> StepResult[FinalTrainingResultDTO]:
        method_id = input_dto.config.model_name
        params = getattr(input_dto.config, "parameters", getattr(input_dto.config, "metadata", {}))

        log.info(f"Finalni trenink: {method_id} (Run: {run_ctx.run_id})")

        x_all, y_all = self._collect_all_data(input_dto)

        pipeline = ModelFactory.create(method_id, params)
        model = GenericSklearnModel(pipeline, method_id)
        model.fit(x_all, y_all)

        train_acc = float(accuracy_score(y_all, model.predict(x_all)))
        history = TrainingHistory(
            train_loss=[1.0 - train_acc],
            train_metrics={"accuracy": [train_acc]},
        )

        trained_model = TrainedModelDTO(
            model=model,
            model_name=method_id,
            history=history,
            best_epoch=0,
            best_validation_metric_name=None,
            best_validation_metric_value=None,
            fold_idx=None,
        )

        log.info(f"Finalni model natrénovan: accuracy={train_acc:.4f} na {len(y_all)} vzorcich")

        return StepResult(FinalTrainingResultDTO(trained_model=trained_model))

    def _collect_all_data(self, input_dto: FinalTrainingInputDTO) -> tuple[np.ndarray, np.ndarray]:
        x_list, y_list = [], []
        for fold in input_dto.folds:
            if fold.test_data:
                x, y = self._extract_data_and_labels(fold.test_data)
                x_list.append(x)
                y_list.append(y)
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

    def _extract_data_and_labels(self, data_dto: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        x_list, y_list = [], []
        for recording in data_dto.data:
            epochs = recording.data
            if hasattr(epochs, "get_data"):
                x_list.append(epochs.get_data(copy=False))
                y_list.append(epochs.events[:, -1])
            else:
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)
