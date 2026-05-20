from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.config.dataset_export_config import DatasetExportConfig
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO


class IDatasetExporter(ABC):
    @abstractmethod
    def run(
        self, config: DatasetExportConfig, data: DatasetSplitDTO, run_ctx: RunContext
    ) -> StepResult[None]:
        """
        Exports the dataset to the specified format.

        Args:
            config: Export configuration.
            data: The dataset splits to export.
            run_ctx: Pipeline execution context.
        """
        raise NotImplementedError
