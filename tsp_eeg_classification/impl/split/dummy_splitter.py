import logging

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO, FoldDTO
from tsp_eeg_classification.types.dto.split.split_input_dto import SplitInputDTO
from tsp_eeg_classification.types.interfaces.splitter import ISplitter


class DummySplitter(ISplitter):
    def run(self, input_dto: SplitInputDTO, run_ctx: RunContext) -> StepResult[DatasetSplitDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy splitter")

        single_fold = FoldDTO(
            fold_idx=0,
            train_data=input_dto.data,
            test_data=None,
        )

        data: DatasetSplitDTO = DatasetSplitDTO(
            folds=[single_fold],
            validation_data=None,
        )
        return StepResult(data)
