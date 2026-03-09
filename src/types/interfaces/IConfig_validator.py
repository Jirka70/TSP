from __future__ import annotations

from abc import ABC, abstractmethod

from src.validation.contracts.validation_input import ValidationInput
from src.validation.contracts.validation_message import ValidationMessage


class IConfigValidator(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def validate(self, input_dto: ValidationInput) -> list[ValidationMessage]:
        raise NotImplementedError
