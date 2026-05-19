from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_export_config import DatasetExportConfig
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO


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
