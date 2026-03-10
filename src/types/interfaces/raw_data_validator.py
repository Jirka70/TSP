from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.validation.contracts.validation_result import ValidationResult


class IRawDataValidator(ABC):
    @abstractmethod
    def validate(self, raw_data: RawDataDTO, run_ctx: RunContext) -> ValidationResult:
        raise NotImplementedError
