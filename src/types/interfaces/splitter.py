from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.split.split_input_dto import SplitInputDTO


class ISplitter(ABC):
    @abstractmethod
    def run(
        self,
        input_dto: SplitInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[DatasetSplitDTO]:
        raise NotImplementedError
