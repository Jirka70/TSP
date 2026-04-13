import moabb.datasets as moabb_datasets
import moabb.paradigms as moabb_paradigms
from moabb.datasets.base import BaseDataset

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.interfaces.data_loader import IDataLoader


class MOABBDataLoader(IDataLoader):
    """
    See datasets:
    https://moabb.neurotechx.com/docs/generated/moabb.datasets.Yang2025.html
    """

    @staticmethod
    def _create_dataset(name: str):
        try:
            data_class = getattr(moabb_datasets, name)
            return data_class()
        except AttributeError:
            raise ValueError(f"Dataset {name} was not found")

    @staticmethod
    def _matches_optional_filter(value: str | int, allowed_values: list[str | int] | None) -> bool:
        if allowed_values is None:
            return True

        value_str = str(value)
        allowed_str = {str(v) for v in allowed_values}
        return value_str in allowed_str

    def _load_raw_recordings(self, dataset: BaseDataset, config: DatasetConfig):
        data = dataset.get_data(subjects=config.subject_ids)

        recordings: list[RecordingDTO] = []

        for subject_id, sessions in data.items():
            for session_id, runs in sessions.items():
                if not self._matches_optional_filter(session_id, config.session_ids):
                    continue

                for run_id, raw in runs.items():
                    if not self._matches_optional_filter(run_id, config.run_ids):
                        continue

                    recordings.append(
                        RecordingDTO(
                            data=raw,
                            dataset_name=config.name,
                            subject_id=subject_id,
                            session_id=str(session_id),
                            run_id=run_id,
                            metadata={
                                "sfreq": raw.info["sfreq"],
                                "n_channels": len(raw.ch_names),
                                "channel_names": list(raw.ch_names),
                            },
                        )
                    )

        return recordings

    @staticmethod
    def _create_paradigm(name: str):
        try:
            paradigm_class = getattr(moabb_paradigms, name)
            return paradigm_class()
        except AttributeError:
            raise ValueError(f"Paradigm {name} was not found")

    def run(self, config: DatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        dataset_name: str = config.name

        dataset = self._create_dataset(dataset_name)
        recordings = self._load_raw_recordings(dataset, config=config)

        res: RawDataDTO = RawDataDTO(recordings)
        return StepResult(res)
