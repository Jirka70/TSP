import moabb.datasets as moabb_datasets
import moabb.paradigms as moabb_paradigms
from moabb.datasets.base import BaseDataset

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
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
        except AttributeError as err:
            raise ValueError(f"Dataset {name} was not found")

    @staticmethod
    def _matches_optional_filter(value: str | int, allowed_values: list[str | int] | None) -> bool:
        if allowed_values is None:
            return True

        value_str = str(value)
        allowed_str = {str(v) for v in allowed_values}
        return value_str in allowed_str

    @staticmethod
    def _get_loadable_subject_ids(dataset: BaseDataset, requested_subject_ids: list[int], log: PipelineLogger) -> list[int]:
        subject_list = getattr(dataset, "subject_list", None)
        if subject_list is None:
            return requested_subject_ids

        available_subject_ids = {str(subject_id) for subject_id in subject_list}
        loadable_subject_ids: list[int] = []

        for subject_id in requested_subject_ids:
            if str(subject_id) in available_subject_ids:
                loadable_subject_ids.append(subject_id)
            else:
                log.warning(f'subject_id "{subject_id}" was not found. Skipping...')

        return loadable_subject_ids

    def _load_raw_recordings(self, dataset: BaseDataset, config: ExternalDatasetConfig, log: PipelineLogger):
        subject_ids = self._get_loadable_subject_ids(dataset, config.subject_ids, log)
        if not subject_ids:
            return []

        data = dataset.get_data(subjects=subject_ids)

        recordings: list[RecordingDTO] = []

        log.info(f"Started loading dataset {config.name}")
        for subject_id, sessions in data.items():
            log.info(f"Started loading subject with id: {subject_id}", VerbosityLevel.DETAILED)
            for session_id, runs in sessions.items():
                log.info(f"Started loading session {session_id} of subject {subject_id}", VerbosityLevel.TRACE)
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
        except AttributeError as err:
            raise ValueError(f"Paradigm {name} was not found")

    def run(self, config: ExternalDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        log = run_ctx.logger.for_step("MOABB_DATA_LOAD")
        log.info("Starting loading MOABB data", VerbosityLevel.QUIET)
        dataset_name: str = config.name

        dataset = self._create_dataset(dataset_name)
        recordings = self._load_raw_recordings(dataset, config=config, log=log)

        res: RawDataDTO = RawDataDTO(recordings)
        return StepResult(res)
