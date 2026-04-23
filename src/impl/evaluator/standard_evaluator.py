import logging

import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.evaluation.fold_evaluation_result_dto import FoldEvaluationResultDTO
from src.types.dto.split.dataset_split_dto import FoldDTO
from src.types.interfaces.evaluator import IEvaluator

log = logging.getLogger(__name__)


class StandardEvaluator(IEvaluator):
    """
    A universal evaluator that works with any IModel implementation.

    It supports both cross-validation and single-model evaluation.
    """

    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> StepResult[EvaluationResultDTO]:
        """Runs the evaluation process for the given models and data.

        Args:
            input_dto (EvaluationInputDTO): DTO containing models and fold data.
            run_ctx (RunContext): Context of the current execution.

        Returns:
            StepResult[EvaluationResultDTO]: The result of the evaluation step.

        Raises:
            ValueError: If no models are provided or no results were generated.
        """
        if not input_dto.trained_models:
            raise ValueError("EvaluationInputDTO does not include any model.")

        fold_results: list[FoldEvaluationResultDTO] = []
        all_y_true = []
        all_y_pred = []
        all_probs = []

        # Determine which metrics to compute
        requested_metrics = input_dto.config.metrics or ["accuracy", "f1_weighted"]

        for model_dto in input_dto.trained_models:
            # Determine which folds to evaluate this model on
            if model_dto.fold_idx is not None:
                relevant_folds = [f for f in input_dto.folds if f.fold_idx == model_dto.fold_idx]
            else:
                relevant_folds = input_dto.folds

            for fold in relevant_folds:
                if not fold.test_data or not fold.test_data.data:
                    continue

                x_test, y_true = self._extract_data(fold)

                # Predict
                y_pred = model_dto.model.predict(x_test)

                # Optional: Predict probabilities if supported
                probs = None
                try:
                    probs = model_dto.model.predict_class_probability(x_test)
                except (AttributeError, NotImplementedError):
                    pass

                # Compute metrics for this fold
                fold_metrics = {}
                for m_name in requested_metrics:
                    scorer = get_scorer(m_name)
                    if hasattr(scorer, "_score_func"):
                        fold_metrics[m_name] = float(scorer._score_func(y_true, y_pred, **scorer._kwargs))
                    else:
                        log.warning(f"Could not calculate metric '{m_name}' directly from labels.")

                fold_res = FoldEvaluationResultDTO(
                    fold_idx=fold.fold_idx if fold.fold_idx is not None else 0,
                    metrics=fold_metrics,
                    predictions=y_pred.tolist(),
                    targets=y_true.tolist(),
                    probabilities=probs.tolist() if probs is not None else None,
                    confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
                )
                fold_results.append(fold_res)

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                if probs is not None:
                    all_probs.extend(probs)

        if not fold_results:
            raise ValueError("No evaluation results were generated. Check if test data is present.")

        # Aggregate metrics
        aggregate_metrics = {}
        for m_name in requested_metrics:
            values = [res.metrics[m_name] for res in fold_results if m_name in res.metrics]
            if values:
                aggregate_metrics[m_name] = float(np.mean(values))
                log.info(f"Aggregate {m_name}: {aggregate_metrics[m_name]:.4f}")

        overall_cm = confusion_matrix(all_y_true, all_y_pred).tolist()

        result = EvaluationResultDTO(
            metrics=aggregate_metrics,
            fold_results=fold_results,
            predictions=all_y_pred,
            targets=all_y_true,
            probabilities=all_probs if all_probs else None,
            confusion_matrix=overall_cm
        )

        return StepResult(result)

    def _extract_data(self, fold: FoldDTO) -> tuple[np.ndarray, np.ndarray]:
        """Extracts X and y from FoldDTO.

        Args:
            fold (FoldDTO): The fold data to extract from.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing feature matrix (X) and labels (y).
        """
        x_list = []
        y_list = []

        for recording in fold.test_data.data:
            epochs = recording.data
            if hasattr(epochs, "get_data"):
                # Handle MNE Epochs
                x_list.append(epochs.get_data(copy=False))
                y_list.append(epochs.events[:, -1])
            else:
                # Handle NumPy arrays
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return x, y
