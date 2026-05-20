from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO


class IAugmentor(ABC):
    @abstractmethod
    def run(
        self, input_dto: AugmentationInputDTO, run_ctx: RunContext
    ) -> StepResult[DatasetSplitDTO]:
        raise NotImplementedError
