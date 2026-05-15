import copy
import dataclasses
import logging
import mne
import mne.io

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.paradigm_config import ParadigmConfig
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from src.types.interfaces.paradigm import IParadigm


class ParadigmPreprocessor(IParadigm):
    """
    Orchestrates the transition from Raw signal to segmented Epochs.
    Supports both a custom MNE-based implementation and native MOABB paradigms.
    """

    def _update_entry_data(self, entry: object, epochs: mne.Epochs) -> object:
        """Safely updates the data field of a DTO, handling frozen dataclasses."""
        if dataclasses.is_dataclass(entry):
            return dataclasses.replace(entry, data=epochs)
        elif hasattr(entry, "_replace"):
            return entry._replace(data=epochs)
        else:
            new_entry = copy.copy(entry)
            new_entry.data = epochs
            return new_entry

    def _normalize_event_name(self, value: object) -> str:
        return str(value).strip().lower().replace(" ", "_")


    def run(self, input_dto: ParadigmInputDTO, run_ctx: RunContext) -> StepResult[ParadigmResultDTO]:
        log = logging.getLogger(__name__)
        cfg: ParadigmConfig = input_dto.paradigm_preprocessing_config

        processed_items = []

        # Unify event parsing (takes key from dictionary or value from list)
        if isinstance(cfg.events, dict):
            configured_events = list(cfg.events.keys())
        elif isinstance(cfg.events, list):
            configured_events = [str(e) for e in cfg.events]
        else:
            configured_events = [str(cfg.events)]

        configured_events_normalized = {self._normalize_event_name(name) for name in configured_events}

        for i, entry in enumerate(input_dto.data.data):
            raw: mne.io.Raw = entry.data

            # Apply bandpass filter using nested filter configuration
            raw.filter(
                l_freq=cfg.filter.fmin,
                h_freq=cfg.filter.fmax,
                fir_design="firwin",
                skip_by_annotation="edge"
            )

            events, event_id = mne.events_from_annotations(raw)
            event_id_filtered = {
                k: v for k, v in event_id.items()
                if self._normalize_event_name(k) in configured_events_normalized or str(v) in configured_events_normalized
            }

            if not event_id_filtered:
                continue

            # Segment Raw data into Epochs using window configuration
            epochs = mne.Epochs(
                raw, events=events, event_id=event_id_filtered,
                tmin=cfg.window.tmin, tmax=cfg.window.tmax,
                baseline=tuple(cfg.window.baseline) if cfg.window.baseline else None,
                reject_by_annotation=cfg.reject_by_annotation,
                preload=cfg.preload,
            )

            if len(epochs) == 0:
                continue

            # Perform resampling if enabled in the configuration
            if cfg.resampling.enabled:
                epochs.resample(cfg.resampling.sfreq)

            processed_items.append(self._update_entry_data(entry, epochs))

        return StepResult(ParadigmResultDTO(data=processed_items))