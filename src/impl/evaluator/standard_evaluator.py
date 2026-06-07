import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator


class StandardEvaluator(IEvaluator):
    """
    A universal evaluator that works with any IModel implementation.

    It evaluates a single trained model on global validation data.
    """

    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> StepResult[EvaluationResultDTO]:
        """
        Runs the evaluation process for a single model using global validation data.

        This method supports evaluating a trained model strictly on the global validation
        data partition. The process follows these sequential stages:
        1. Initialization: Validates models and ensures global validation data is available.
        2. Data Extraction: Prepares the validation feature matrix and true labels.
        3. Model Evaluation: Generates predictions and class probabilities using the model.
        4. Metrics Computation: Calculates requested performance metrics and confusion matrix.

        Args:
            input_dto (EvaluationInputDTO): DTO containing models and split data.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[EvaluationResultDTO]: The evaluation results on the validation set.

        Raises:
            ValueError: If no models are provided or if global validation data is missing.
        """
        log = run_ctx.logger.for_step("STANDART_EVALUATION")

        # --- 1. Initialization ---
        if not input_dto.trained_models:
            log.info("EvaluationInputDTO does not include any model. Skipping evaluation.", VerbosityLevel.NORMAL)
            return StepResult(EvaluationResultDTO(
                metrics={},
                predictions=[],
                targets=[],
                probabilities=None,
                confusion_matrix=None
            ))

        # We evaluate only the primary/first model
        model_dto = input_dto.trained_models[0]

        # Strictly use global validation data
        validation_data = input_dto.dataset_split.validation_data if input_dto.dataset_split else None

        if not validation_data or not validation_data.data:
            log.error("No global validation data found. Evaluation strictly requires 'validation_data' partition.")
            raise ValueError("No global validation data found. Evaluation strictly requires 'validation_data' partition.")

        log.info(f"Starting evaluation for model '{model_dto.model_name}' on global validation data", VerbosityLevel.QUIET)

        # --- 2. Data Extraction ---
        x_val, y_true = self.extract_data(validation_data)

        # --- 3. Model Evaluation ---
        log.info(f"Evaluating model...", VerbosityLevel.DETAILED)
        
        # Predict labels
        y_pred = model_dto.model.predict(x_val)

        # Predict probabilities if supported by the model
        probabilities = None
        try:
            probabilities = model_dto.model.predict_class_probability(x_val)
        except (AttributeError, NotImplementedError):
            pass

        # --- 4. Metrics Computation ---
        model_metrics = {}
        for m_name in input_dto.config.metrics:
            scorer = get_scorer(m_name)
            if hasattr(scorer, "_score_func"):
                val = float(scorer._score_func(y_true, y_pred, **scorer._kwargs))
                model_metrics[m_name] = val
                log.info(f"Metric {m_name}: {val:.4f}", VerbosityLevel.TRACE)
            else:
                log.warning(f"Could not calculate metric '{m_name}' directly from labels.")

        overall_cm = confusion_matrix(y_true, y_pred).tolist()

        result = EvaluationResultDTO(
            metrics=model_metrics,
            predictions=y_pred.tolist(),
            targets=y_true.tolist(),
            probabilities=probabilities.tolist() if probabilities is not None else None,
            confusion_matrix=overall_cm
        )

        log.info("Evaluation completed successfully", VerbosityLevel.QUIET)
        return StepResult(result)


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
