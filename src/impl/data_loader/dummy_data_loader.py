import logging

from moabb.datasets import BNCI2014_001

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDto
from src.types.interfaces.data_loader import IDataLoader


class DummyLoader(IDataLoader):
    def run(self, input_dto: DatasetConfig, run_ctx: RunContext) -> StepResult[RawPreprocessingInputDto]:
        log = logging.getLogger(__name__)
        log.info("Loading MOABB dataset as mne.Raw")

        dataset = BNCI2014_001()
        subject_id = 1

        data_dict = dataset.get_data(subjects=[subject_id])

        sessions = data_dict[subject_id]
        first_session_key = next(iter(sessions))
        runs = sessions[first_session_key]
        first_run_key = next(iter(runs))

        mne_raw = runs[first_run_key]

        result_dto = RawPreprocessingInputDto(raw_preprocessing_config=None, signal=mne_raw)

        return StepResult(result_dto, None, [])
