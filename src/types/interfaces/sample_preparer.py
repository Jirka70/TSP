from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.model.prepared_samples_dto import PreparedSamplesDTO
from src.types.dto.model.sample_preparation_input_dto import SamplePreparationInputDTO


class ISamplePreparer(ABC):
    @abstractmethod
    def run(self, input_dto: SamplePreparationInputDTO, run_context: RunContext) -> PreparedSamplesDTO:
        raise NotImplementedError
