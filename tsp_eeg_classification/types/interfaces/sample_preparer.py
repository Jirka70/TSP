from abc import ABC, abstractmethod

from tsp_eeg_classification.types.dto.model.prepared_samples_dto import PreparedSamplesDTO
from tsp_eeg_classification.types.dto.model.sample_preparation_input_dto import SamplePreparationInputDTO

from tsp_eeg_classification.pipeline.context.run_context import RunContext


class ISamplePreparer(ABC):
    @abstractmethod
    def run(
        self, input_dto: SamplePreparationInputDTO, run_context: RunContext
    ) -> PreparedSamplesDTO:
        raise NotImplementedError
    