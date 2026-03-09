from abc import ABC, abstractmethod

from src.types.dto.training.prepared_samples_dto import PreparedSamplesDTO
from src.types.dto.training.sample_preparation_input_dto import SamplePreparationInputDTO


class ISamplePreparer(ABC):
    @abstractmethod
    def run(self, input_dto: SamplePreparationInputDTO) -> PreparedSamplesDTO:
        raise NotImplementedError