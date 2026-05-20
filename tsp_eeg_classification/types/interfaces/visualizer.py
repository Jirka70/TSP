from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from tsp_eeg_classification.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from tsp_eeg_classification.types.dto.raw_augmentation.raw_augmented_dto import RawAugmentedDTO
from tsp_eeg_classification.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO


class IVisualizer(ABC):
    """Unified interface for all visualization backends."""

    @abstractmethod
    def visualize_raw(self, data: RawPreprocessedDTO, run_ctx: RunContext) -> None:
        """Visualizes raw preprocessed data (e.g., PSD, signal traces)."""
        raise NotImplementedError

    @abstractmethod
    def visualize_raw_augmentation(self, data: RawAugmentedDTO, run_ctx: RunContext, copies_per_sample: int = 0) -> None:
        """Visualizes raw augmented data."""
        raise NotImplementedError

    @abstractmethod
    def visualize_epochs(self, data: EpochPreprocessedDTO, run_ctx: RunContext) -> None:
        """Visualizes epoched data (e.g., ERPs, time-frequency)."""
        raise NotImplementedError

    @abstractmethod
    def visualize_augmentation(self, data: DatasetSplitDTO, run_ctx: RunContext, copies_per_sample: int = 0) -> None:
        """Visualizes augmented data (e.g., comparison of original and augmented samples)."""
        raise NotImplementedError

    @abstractmethod
    def visualize_evaluation(self, data: EvaluationResultDTO, run_ctx: RunContext, model_name: str) -> None:
        """Visualizes evaluation results (e.g., confusion matrix, metrics)."""
        raise NotImplementedError
