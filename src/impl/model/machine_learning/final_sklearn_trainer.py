import hashlib
from typing import List, Any, Tuple, Dict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.impl.model.machine_learning.generic_sklearn_model import GenericSklearnModel
from src.impl.model.model_factory import ModelFactory
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.train_history import TrainingHistory
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer


class FinalSklearnTrainer(IFinalTrainer):
    """
    Train a Scikit-learn model on all available folds for final export.

    The training set is the union of all fold training partitions, deduplicated and with every
    sample of the global validation set removed, because the final model is evaluated on it.

    Note:
        `input_dto.training_data` is intentionally NOT used. It holds the data before splitting,
        including the global validation partition, so training on it would leak validation data
        into the final model.
    """

    def run(self, input_dto: FinalTrainingInputDTO, run_ctx: RunContext) -> StepResult[FinalTrainingResultDTO]:
        """
        Fit the configured model on all available training data and wrap the result.

        Args:
            input_dto (FinalTrainingInputDTO): Input data container containing folds and configuration.
            run_ctx (RunContext): Context information for the current execution run.

        Returns:
            StepResult[FinalTrainingResultDTO]: Encapsulated final trained model artifact.
        """
        log: PipelineLogger = run_ctx.logger.for_step("FINAL_TRAINING")
        method_id = input_dto.config.model_name
        params = getattr(input_dto.config, "parameters", getattr(input_dto.config, "metadata", {}))

        log.info(f"Final training: {method_id} (Run: {run_ctx.run_id})", VerbosityLevel.QUIET)

        # Accumulate all training samples from the provided folds
        x_all, y_all = self._collect_all_data(input_dto, log)

        pipeline : Pipeline = ModelFactory.create(method_id, params)
        model : GenericSklearnModel = GenericSklearnModel(pipeline, method_id)
        model.fit(x_all, y_all)

        train_acc : float = float(accuracy_score(y_all, model.predict(x_all)))
        history = TrainingHistory(
            train_loss=[1.0 - train_acc],
            train_metrics={"accuracy": [train_acc]},
        )

        trained_model : TrainedModelDTO = TrainedModelDTO(
            model=model,
            model_name=method_id,
            history=history,
            best_epoch=0,
            best_validation_metric_name=None,
            best_validation_metric_value=None,
            fold_idx=None,
        )

        log.info(f"Final model trained: accuracy={train_acc:.4f} on {len(y_all)} samples", VerbosityLevel.QUIET)

        return StepResult(FinalTrainingResultDTO(trained_model=trained_model))

    def _collect_all_data(self, input_dto: FinalTrainingInputDTO, log: PipelineLogger) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect training samples and labels across all folds for the final model.

        Samples present in more than one fold are kept only once (first occurrence). Samples that
        are also part of the global validation set are excluded, because the final model is
        evaluated on that set. Augmented copies are distinct samples and are kept.

        Args:
            input_dto (FinalTrainingInputDTO): Input container with fold and validation data.
            log (PipelineLogger): Logger of the final training step.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Concatenated feature matrix (X) and labels (y).

        Raises:
            ValueError: If the same sample occurs with different labels.
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for fold in input_dto.folds:
            if fold.train_data:
                x, y = self._extract_data_and_labels(fold.train_data)
                x_list.append(x)
                y_list.append(y)

        if not x_list:
            return np.array([]), np.array([])

        x_concat = np.concatenate(x_list, axis=0)
        y_concat = np.concatenate(y_list, axis=0)

        validation_keys: set[bytes] = set()
        if input_dto.validation_data and input_dto.validation_data.data:
            x_val, _ = self._extract_data_and_labels(input_dto.validation_data)
            validation_keys = {self._sample_key(sample) for sample in x_val}

        seen_labels: Dict[bytes, Any] = {}
        keep_indices: List[int] = []
        duplicate_count = 0
        validation_overlap_count = 0

        for idx, sample in enumerate(x_concat):
            key = self._sample_key(sample)

            if key in seen_labels:
                if seen_labels[key] != y_concat[idx]:
                    raise ValueError(f"Sample at index {idx} occurs in the training folds with conflicting labels ({seen_labels[key]} vs {y_concat[idx]}).")
                duplicate_count += 1
                continue
            seen_labels[key] = y_concat[idx]

            if key in validation_keys:
                validation_overlap_count += 1
                continue

            keep_indices.append(idx)

        log.info(f"Removed {duplicate_count} duplicate samples shared between folds.", VerbosityLevel.DETAILED)
        if validation_overlap_count > 0:
            log.warning(f"Excluded {validation_overlap_count} training samples that are also part of the global validation set.")

        return x_concat[keep_indices], y_concat[keep_indices]

    @staticmethod
    def _sample_key(sample: np.ndarray) -> bytes:
        """
        Build an exact-match identity key of a single sample.

        Args:
            sample (np.ndarray): A single epoch or feature vector.

        Returns:
            bytes: Digest of the sample's shape and float64 values.
        """
        data = np.ascontiguousarray(sample, dtype=np.float64)
        return hashlib.sha1(str(data.shape).encode() + data.tobytes()).digest()

    def _extract_data_and_labels(self, data_dto: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from preprocessed epoch recordings.

        Args:
            data_dto (EpochPreprocessedDTO): Preprocessed data container.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and labels arrays.
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for recording in data_dto.data:
            epochs : Any = recording.data

            if hasattr(epochs, "get_data"):
                x_list.append(epochs.get_data(copy=False))
                y_list.append(epochs.events[:, -1])
            else:
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))

        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)
