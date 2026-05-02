import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import confusion_matrix, get_scorer

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.evaluation.fold_evaluation_result_dto import FoldEvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator

log = logging.getLogger(__name__)


class StandardEvaluator(IEvaluator):
    """
    A universal evaluator that works with any IModel implementation.

    It evaluates a single trained model on global validation data.
    """

    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> StepResult[EvaluationResultDTO]:
        """Runs the evaluation process for the provided model and validation data.

        Args:
            input_dto (EvaluationInputDTO): DTO containing the model and split data.
            run_ctx (RunContext): Context of the current execution.

        Returns:
            StepResult[EvaluationResultDTO]: The result of the evaluation step.

        Raises:
            ValueError: If no model is provided or validation data is missing.
        """
        if not input_dto.trained_models:
            raise ValueError("EvaluationInputDTO does not include any model.")

        #  There is always only one model at this stage phase of pipeline
        model_dto = input_dto.trained_models[0]

        validation_data = input_dto.dataset_split.validation_data
        if not validation_data or not validation_data.data:
            raise ValueError("No validation data found in DatasetSplitDTO. Evaluation requires global validation data.")

        log.info(f"Starting evaluation for model: {model_dto.model_name}")

        # Extract validation data
        x_val, y_true = self.extract_data(validation_data)

        # Predict
        y_pred = model_dto.model.predict(x_val)

        # Optional: Predict probabilities if supported
        probabilities = None
        try:
            probabilities = model_dto.model.predict_class_probability(x_val)
        except (AttributeError, NotImplementedError):
            pass

        # Compute metrics
        requested_metrics = input_dto.config.metrics
        model_metrics = {}
        for m_name in requested_metrics:
            scorer = get_scorer(m_name)
            if hasattr(scorer, "_score_func"):
                val = float(scorer._score_func(y_true, y_pred, **scorer._kwargs))
                model_metrics[m_name] = val
                log.info(f"Metric {m_name}: {val:.4f}")
            else:
                log.warning(f"Could not calculate metric '{m_name}' directly from labels.")

        overall_cm = confusion_matrix(y_true, y_pred).tolist()

        # Create the single result entry
        fold_res = FoldEvaluationResultDTO(
            fold_idx=model_dto.fold_idx if model_dto.fold_idx is not None else 0,
            metrics=model_metrics,
            predictions=y_pred.tolist(),
            targets=y_true.tolist(),
            probabilities=probabilities.tolist() if probabilities is not None else None,
            confusion_matrix=overall_cm,
        )

        # Visualization
        self.visualize_results(y_true, y_pred, overall_cm, model_dto.model_name, model_metrics)

        result = EvaluationResultDTO(metrics=model_metrics, fold_results=[fold_res], predictions=y_pred.tolist(), targets=y_true.tolist(), probabilities=probabilities.tolist() if probabilities is not None else None, confusion_matrix=overall_cm)

        return StepResult(result)

    def visualize_results(self, y_true: np.ndarray, y_pred: np.ndarray, cm: list[list[int]], model_name: str, metrics: dict[str, float]) -> None:
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

        # 2. Class distribution
        plt.subplot(1, 3, 2)
        classes, counts_true = np.unique(y_true, return_counts=True)
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
        plt.title("Evaluation Metrics")
        plt.xlabel("Value")

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontweight="bold")

        plt.tight_layout()

        output_dir = Path(HydraConfig.get().runtime.output_dir).absolute()
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        filename = f"evaluation_{model_name.lower().replace(' ', '_')}.png"
        save_path = plots_dir / filename

        plt.savefig(str(save_path))
        log.info(f"Evaluation plot saved to: {save_path}")

        plt.show()
        plt.close()

    def extract_data(self, preprocessed_data: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts and concatenates the feature matrix (X) and labels (y) from a preprocessed dataset.

        This method iterates through the recordings in the provided data transfer object.
        It gracefully handles both MNE Epochs objects (extracting data and event labels natively)
        and standard NumPy arrays (extracting labels from the recording's metadata).

        Args:
            preprocessed_data (EpochPreprocessedDTO): The data transfer object containing
                a list of preprocessed recordings.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two elements:
                - X (np.ndarray): The combined feature matrix concatenated along the first axis.
                - y (np.ndarray): The combined 1D array of labels corresponding to the features.
        """
        x_list = []
        y_list = []

        for recording in preprocessed_data.data:
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
