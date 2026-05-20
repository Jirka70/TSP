import copy
import dataclasses
import logging
from typing import Any, Dict, List, Set, Union

import mne
import mne.io
import numpy as np

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.paradigm_config import ParadigmConfig
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from src.types.interfaces.paradigm import IParadigm


class ParadigmPreprocessor(IParadigm):
    """
    Orchestrates the transition from Raw signal to segmented Epochs.
    Supports custom MNE-based implementation.
    """

    def _update_entry_data(self, entry: Any, epochs: mne.Epochs) -> Any:
        """Safely updates the data field of a DTO, handling frozen dataclasses."""
        if dataclasses.is_dataclass(entry):
            return dataclasses.replace(entry, data=epochs)
        elif hasattr(entry, "_replace"):
            return entry._replace(data=epochs)
        else:
            new_entry = copy.copy(entry)
            new_entry.data = epochs
            return new_entry

    def _normalize_event_name(self, value: Any) -> str:
        """Normalizes event names by converting to lowercase, stripping whitespace, and replacing spaces with underscores."""
        return str(value).strip().lower().replace(" ", "_")


    def run(self, input_dto: ParadigmInputDTO, run_ctx: RunContext) -> StepResult[ParadigmResultDTO]:
        """
        Processes raw MNE data by filtering, segmenting into epochs, and optionally resampling.

        Args:
            input_dto (ParadigmInputDTO): The input data transfer object containing
                the raw MNE data and the paradigm preprocessing configuration.
            run_ctx (RunContext): The execution context for the current pipeline run.

        Returns:
            StepResult[ParadigmResultDTO]: A step result container holding the
                processed, epoched, and optionally resampled data entries.
        """
        log: logging.Logger = logging.getLogger(__name__)
        config: ParadigmConfig = input_dto.paradigm_preprocessing_config

        processed_items: List[Any] = []

        # Unify event parsing (takes key from dictionary or value from list)
        configured_events: List[str]
        if isinstance(config.events, dict):
            configured_events = list(config.events.keys())
        elif isinstance(config.events, list):
            configured_events = [str(e) for e in config.events]
        else:
            configured_events = [str(config.events)]

        configured_events_normalized: Set[str] = {self._normalize_event_name(name) for name in configured_events}

        log.info(f"Starting paradigm preprocessing for {len(input_dto.data.data)} entries with configured events: {configured_events_normalized}")

        for i, entry in enumerate(input_dto.data.data):
            raw: mne.io.Raw = entry.data

            # Apply bandpass filter using nested filter configuration
            raw.filter(
                l_freq=config.filter.fmin,
                h_freq=config.filter.fmax,
                fir_design="firwin",
                skip_by_annotation="edge"
            )

            # mne.events_from_annotations returns an (N, 3) int array and a mapping dict
            events: np.ndarray
            event_id: Dict[str, int]
            events, event_id = mne.events_from_annotations(raw)

            event_id_filtered: Dict[str, int] = {
                k: v for k, v in event_id.items()
                if self._normalize_event_name(k) in configured_events_normalized or str(v) in configured_events_normalized
            }

            if not event_id_filtered:
                continue

            # Segment Raw data into Epochs
            epochs: mne.Epochs = mne.Epochs(
                raw, events=events, event_id=event_id_filtered,
                tmin=config.window.tmin, tmax=config.window.tmax,
                baseline=tuple(config.window.baseline) if config.window.baseline else None,
                reject_by_annotation=config.reject_by_annotation,
                preload=config.preload,
            )

            if len(epochs) == 0:
                continue

            if config.resampling.enabled:
                epochs.resample(config.resampling.sfreq)

            processed_items.append(self._update_entry_data(entry, epochs))

        return StepResult(ParadigmResultDTO(data=processed_items))