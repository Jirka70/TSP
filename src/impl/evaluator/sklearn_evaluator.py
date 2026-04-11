import logging

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator

log = logging.getLogger(__name__)


class SklearnEvaluator(IEvaluator):
    """
    Concrete implementation of the IEvaluator interface using Scikit-learn metrics.

    This evaluator calculates standard performance metrics by comparing model predictions
    against ground truth labels provided in the test dataset.
    """

    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> StepResult[EvaluationResultDTO]:
        """
        Executes the evaluation process on the provided test dataset.

        The method performs the following steps:
        1. Validates the presence of test data.
        2. Generates predictions using the trained model's inference method.
        3. Computes Accuracy, Weighted F1-Score, and a Confusion Matrix.
        4. Encapsulates the results into a serializable EvaluationResultDTO.

        Args:
            input_dto (EvaluationInputDTO): DTO containing the trained model instance,
                test data, and evaluation configuration.
            run_ctx (RunContext): Execution context providing metadata for the current pipeline run.

        Returns:
            StepResult[EvaluationResultDTO]: A standardized step result containing a dictionary
                of calculated metrics.

        Raises:
            ValueError: If the input_dto does not contain valid test_data.
        """
        log.info(f"Running evaluation for model: {input_dto.trained_model.model_name}")

        # Collect test data and labels
        if input_dto.test_data is None:
            raise ValueError("EvaluationInputDTO must contain test_data for evaluation.")

        x_test = input_dto.test_data.signal
        y_true = np.asarray(input_dto.test_data.labels)

        # Prediction with IModel interface (using trained model)
        model = input_dto.trained_model.model
        y_pred = model.predict(x_test)

        # Metrics calculation
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred).tolist()

        log.info(f"Evaluation Results -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # Result DTO
        metrics = {"accuracy": float(acc), "f1_score": float(f1), "confusion_matrix": cm, "n_samples": len(y_true)}

        return StepResult(EvaluationResultDTO(metrics))
