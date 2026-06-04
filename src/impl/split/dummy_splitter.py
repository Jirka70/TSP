import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.splitter import ISplitter
from src.types.dto.config.logging.verbosity_level import VerbosityLevel


class DummySplitter(ISplitter):
    def run(self, input_dto: SplitInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        log = run_ctx.logger.for_step("DUMMY_SPLITTER")

        log.info("Running DummySplitter: returning all data as a single fold without actual splitting.", VerbosityLevel.QUIET)
        single_fold = FoldDTO(
            fold_idx=0,
            train_data=input_dto.data,
            test_data=None,
        )

        data: DatasetSplitDTO = DatasetSplitDTO(
            folds=[single_fold],
            validation_data=None,
        )

        log.info("DummySplitter completed. All data is assigned to a single training fold.", VerbosityLevel.QUIET)
        return StepResult(data)
