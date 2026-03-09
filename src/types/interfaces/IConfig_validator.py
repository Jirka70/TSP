# src/types/interfaces/validator.py

from __future__ import annotations

from abc import ABC, abstractmethod

from src.pipeline.contracts.step_result import StepResult
from src.validation.validation_input import ValidationInput


class IConfigValidator(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def validate(self, input_dto: ValidationInput) -> StepResult[ValidatedConfig]:
        raise NotImplementedError