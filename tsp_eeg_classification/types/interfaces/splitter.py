from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO
from tsp_eeg_classification.types.dto.split.split_input_dto import SplitInputDTO


class ISplitter(ABC):
    @abstractmethod
    def run(
        self,
        input_dto: SplitInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[DatasetSplitDTO]:
        raise NotImplementedError
