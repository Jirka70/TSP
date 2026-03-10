from src.pipeline.context.run_context import RunContext
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.raw_data_validator import IRawDataValidator
from src.validation.contracts.validation_result import ValidationResult


class RawDataValidationStage:
    def __init__(self, validator: IRawDataValidator) -> None:
        self._validator = validator

    def run(self, raw_data: RawDataDTO, run_ctx: RunContext) -> ValidationResult:
        return self._validator.validate(raw_data, run_ctx)