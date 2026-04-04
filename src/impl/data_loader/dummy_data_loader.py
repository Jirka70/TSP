import logging

from moabb.datasets import BNCI2014_001

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class DummyLoader(IDataLoader):
    def run(self, input_dto: DatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
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

        result_dto = RawDataDTO(
            signal=mne_raw,
            sampling_freq=mne_raw.info["sfreq"],
            channel_names=mne_raw.ch_names,
        )

        return StepResult(result_dto, None, [])
