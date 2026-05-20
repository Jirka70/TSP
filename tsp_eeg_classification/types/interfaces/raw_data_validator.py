from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.types.dto.load.raw_data_dto import RawDataDTO
from validation.validation_message import ValidationResult


class IRawDataValidator(ABC):
    @abstractmethod
    def validate(self, raw_data: RawDataDTO, run_ctx: RunContext) -> ValidationResult:
        raise NotImplementedError
