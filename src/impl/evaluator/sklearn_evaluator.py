import logging

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator

log = logging.getLogger(__name__)


class SklearnEvaluator(IEvaluator):
    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> StepResult[EvaluationResultDTO]:
        if not input_dto.trained_models:
            raise ValueError("EvaluationInputDTO neobsahuje zadny model.")

        model = input_dto.trained_models[0].model
        model_name = input_dto.trained_models[0].model_name

        x_list, y_list = [], []
        for fold in input_dto.folds:
            if fold.test_data:
                for recording in fold.test_data.data:
                    epochs = recording.data
                    if hasattr(epochs, "get_data"):
                        x_list.append(epochs.get_data(copy=False))
                        y_list.append(epochs.events[:, -1])
                    else:
                        x_list.append(epochs)
                        y_list.append(np.array(recording.metadata.get("labels", [])))

        if not x_list:
            raise ValueError("Zadna testovaci data nebyla nalezena v foldech.")

        x_test = np.concatenate(x_list, axis=0)
        y_true = np.concatenate(y_list, axis=0)

        y_pred = model.predict(x_test)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred).tolist()

        log.info(f"Evaluace [{model_name}] -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

        self._visualize_results(y_true, y_pred, cm, model_name)

        metrics = {"accuracy": float(acc), "f1_score": float(f1), "n_samples": len(y_true)}

        return StepResult(EvaluationResultDTO(metrics=metrics, confusion_matrix=cm))

    def _visualize_results(self, y_true, y_pred, cm, model_name):
        """Vytvoří jednoduchý dashboard s výsledky."""
        plt.figure(figsize=(12, 5))

        # 1. Confusion Matrix Heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predikovaná třída")
        plt.ylabel("Skutečná třída")

        # 2. Porovnání distribuce (Realita vs Predikce)
        plt.subplot(1, 2, 2)
        classes, counts_true = np.unique(y_true, return_counts=True)
        _, counts_pred = np.unique(y_pred, return_counts=True)

        x = np.arange(len(classes))
        width = 0.35
        plt.bar(x - width / 2, counts_true, width, label="Skutečnost", color="gray", alpha=0.6)
        plt.bar(x + width / 2, counts_pred, width, label="Predikce", color="skyblue")

        plt.title("Distribuce tříd")
        plt.xlabel("Třída")
        plt.ylabel("Počet vzorků")
        plt.xticks(x, classes)
        plt.legend()

        plt.tight_layout()
        plt.show()  # Nebo plt.savefig(f"eval_{model_name}.png")
