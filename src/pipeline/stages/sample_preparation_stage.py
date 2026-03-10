from src.pipeline.context.run_context import RunContext
from src.types.dto.training.prepared_samples_dto import PreparedSamplesDTO
from src.types.dto.training.sample_preparation_input_dto import SamplePreparationInputDTO
from src.types.interfaces.sample_preparer import ISamplePreparer


class SamplePreparationStage:
    def __init__(self, sample_preparer: ISamplePreparer) -> None:
        self._sample_preparer = sample_preparer

    def run(self, input_dto: SamplePreparationInputDTO, run_ctx: RunContext) -> PreparedSamplesDTO:
        return self._sample_preparer.run(input_dto, run_ctx)