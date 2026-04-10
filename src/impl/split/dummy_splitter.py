import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.splitter import ISplitter


class DummySplitter(ISplitter):
    def run(
        self, input_dto: SplitInputDTO, run_ctx: RunContext
    ) -> StepResult[DatasetSplitDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy splitter")
        data: DatasetSplitDTO = DatasetSplitDTO(
            train_data=input_dto.data,
            validation_data=None,
            test_data=None,
        )
        return StepResult(data)
