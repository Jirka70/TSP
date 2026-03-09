from abc import ABC, abstractmethod

from src.types.dto.load.raw_data_dto import RawDataDTO
from src.validation.contracts.validation_result import ValidationResult


class IRawDataValidator(ABC):
    @abstractmethod
    def validate(self, raw_data: RawDataDTO) -> ValidationResult:
        raise NotImplementedError
