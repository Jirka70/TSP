import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.hydra_config import HydraConfig
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
                    # Note: Using private _score_func and _kwargs from Scorer objects
                    # to calculate metrics directly from y_true and y_pred.
                    # The metric name is already validated by EvaluationConfig.
                    scorer = get_scorer(m_name)
                    if hasattr(scorer, "_score_func"):
                        fold_metrics[m_name] = float(scorer._score_func(y_true, y_pred, **scorer._kwargs))
                    else:
                        # Fallback for some custom scorers that might not have _score_func
                        # In this project, most will be standard classification metrics.
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

        # Visualization
        self._visualize_results(all_y_true, all_y_pred, overall_cm, input_dto.trained_models[0].model_name if len(input_dto.trained_models) == 1 else "Combined Models", aggregate_metrics)

        result = EvaluationResultDTO(metrics=aggregate_metrics, fold_results=fold_results, predictions=all_y_pred, targets=all_y_true, probabilities=all_probs if all_probs else None, confusion_matrix=overall_cm)

        return StepResult(result)

    def _visualize_results(self, y_true: np.ndarray, y_pred: np.ndarray, cm: list[list[int]], model_name: str, metrics: dict[str, float]) -> None:
        """Creates a simple dashboard with results and saves it to the run directory.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.
            cm (list[list[int]]): Confusion matrix.
            model_name (str): Name of the model for titles and filenames.
            metrics (dict[str, float]): Dictionary of aggregate metrics.
        """
        plt.figure(figsize=(15, 6))

        # 1. Confusion Matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")

        # 2. Class distribution (Reality vs Predicted)
        plt.subplot(1, 3, 2)
        classes, counts_true = np.unique(y_true, return_counts=True)
        # Handle cases where some classes might not be predicted
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        pred_counts_dict = dict(zip(pred_unique, pred_counts, strict=True))
        counts_pred = [pred_counts_dict.get(cls, 0) for cls in classes]

        x = np.arange(len(classes))
        width = 0.35
        plt.bar(x - width / 2, counts_true, width, label="Reality", color="gray", alpha=0.6)
        plt.bar(x + width / 2, counts_pred, width, label="Predicted", color="skyblue")

        plt.title("Class distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.xticks(x, classes)
        plt.legend()

        # 3. Metrics Summary
        plt.subplot(1, 3, 3)
        m_names = list(metrics.keys())
        m_values = [metrics[name] for name in m_names]

        bars = plt.barh(m_names, m_values, color="salmon")
        plt.xlim(0, 1.1)
        plt.title("Aggregate Metrics")
        plt.xlabel("Value")

        # Add values to the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontweight="bold")

        plt.tight_layout()

        # Save the plot
        # Try to get the output directory from Hydra, otherwise fall back to current working directory
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir).absolute()
        except (ValueError, KeyError, RuntimeError):
            # Fallback if not running through Hydra
            output_dir = Path(os.getcwd()).absolute()

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        filename = f"evaluation_{model_name.lower().replace(' ', '_')}.png"
        save_path = plots_dir / filename

        plt.savefig(str(save_path))
        log.info(f"Evaluation plot saved to: {save_path}")

        plt.close()  # Important to free memory

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
