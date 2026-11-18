from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_export_config import DatasetExportConfig
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.dataset_exporter import IDatasetExporter


class DummyDatasetExporter(IDatasetExporter):
    """
    A no-op dataset exporter.
    Used when dataset export is disabled (backend: 'none').
    Does not save any dataset files.
    """

    def run(self, config: DatasetExportConfig, data: DatasetSplitDTO, run_ctx: RunContext) -> StepResult[None]:
        log = run_ctx.logger.for_step("DATASET_EXPORT")
        log.info("Dataset export is set to 'none' - skipping dataset export.")
        return StepResult(None)
