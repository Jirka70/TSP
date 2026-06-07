import copy
import dataclasses
import logging
from typing import Any, List, Optional

import mne
import mne.io

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO
from src.types.interfaces.raw_preprocessing import IRawPreprocessing


class RawPreprocessor(IRawPreprocessing):
    """
    Implements a modular preprocessing strategy for mne.io.Raw EEG data.

    This implementation allows toggling individual transformation steps via
    the configuration 'enabled' flag and includes advanced cleaning techniques
    like ICA, Resampling, and flexible Re-referencing.
    """

    def run(self, input_dto: RawPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[RawPreprocessedDTO]:
        log: PipelineLogger = run_ctx.logger.for_step("RAW_PREPROCESSING")
        config: RawPreprocessingConfig = input_dto.raw_preprocessing_config

        log.info(f"Starting processing with backend: {getattr(config, 'backend', 'default')}", VerbosityLevel.QUIET)
        log.info(f"Processing {len(input_dto.data.data)} continuous EEG recordings", VerbosityLevel.QUIET)

        processed_items: List[Any] = []

        for i, entry in enumerate(input_dto.data.data):
            log.info(f"Processing recording index: {i}", VerbosityLevel.DETAILED)

            raw_copy: mne.io.Raw = entry.data.copy()

            # --- 1. Resampling ---
            if getattr(config.resampling, "enabled", False):
                log.info(f"Resampling signal to {config.resampling.sfreq} Hz", VerbosityLevel.TRACE)
                raw_copy.resample(sfreq=config.resampling.sfreq)

            # --- 2. Bad Channels Identification and Interpolation ---
            if getattr(config.bad_channels_interpolation, "enabled", False):
                if raw_copy.info["bads"]:
                    log.info(f"Interpolating bad channels at index {i}: {raw_copy.info['bads']}", VerbosityLevel.TRACE)
                    raw_copy.interpolate_bads(reset_bads=True)
                else:
                    log.info(f"No bad channels detected for interpolation at index {i}", VerbosityLevel.TRACE)

            # --- 3. Frequency Filtering (High-pass, Low-pass, Notch) ---
            high_pass_filter_enabled: bool = getattr(config.high_pass_filter, "enabled", False)
            low_pass_filter_enabled: bool = getattr(config.low_pass_filter, "enabled", False)

            if high_pass_filter_enabled or low_pass_filter_enabled:
                low_freq: Optional[float] = config.high_pass_filter.l_freq if high_pass_filter_enabled else None
                high_freq: Optional[float] = config.low_pass_filter.h_freq if low_pass_filter_enabled else None
                log.info(f"Applying filter: HPF={low_freq} Hz, LPF={high_freq} Hz", VerbosityLevel.TRACE)
                raw_copy.filter(
                    l_freq=low_freq,
                    h_freq=high_freq,
                    fir_design="firwin",
                    skip_by_annotation="edge"
                )

            if getattr(config.notch_filter, "enabled", False):
                log.info(f"Applying Notch filter: {list(config.notch_filter.freqs)} Hz", VerbosityLevel.TRACE)
                raw_copy.notch_filter(freqs=list(config.notch_filter.freqs), fir_design="firwin")

            # --- 4. ICA (Artifact Rejection) ---
            if getattr(config.ica, "enabled", False):
                log.info(f"Running ICA decomposition (method: {config.ica.method})", VerbosityLevel.TRACE)
                ica: mne.preprocessing.ICA = mne.preprocessing.ICA(
                    n_components=config.ica.n_components,
                    method=config.ica.method,
                    random_state=42
                )
                ica.fit(raw_copy)
                log.info(f"Applying ICA to remove artifact components", VerbosityLevel.TRACE)
                ica.apply(raw_copy)

            # --- 5. Spatial Transformation / Re-referencing ---
            if getattr(config.re_referencing, "enabled", False):
                re_referencing_method: str = config.re_referencing.method.upper()

                if re_referencing_method == "CSD":
                    log.info("Computing Current Source Density (CSD)", VerbosityLevel.TRACE)
                    try:
                        raw_copy = mne.preprocessing.compute_current_source_density(raw_copy)
                    except (RuntimeError, ValueError) as e:
                        log.warning(f"CSD skipped at index {i}: {e}")

                elif re_referencing_method == "AVERAGE":
                    log.info("Applying Common Average Reference (CAR)", VerbosityLevel.TRACE)
                    raw_copy.set_eeg_reference(ref_channels="average")

            # --- 6. Automatic Annotation of Artifacts/Breaks ---
            if getattr(config.annotate_break, "enabled", False):
                log.info("Generating break annotations", VerbosityLevel.TRACE)
                new_annotations: mne.Annotations = mne.preprocessing.annotate_break(
                    raw_copy,
                    min_break_duration=config.annotate_break.min_break_duration,
                    t_start_after_previous=1.0,
                    t_stop_before_next=1.0,
                )
                raw_copy.set_annotations(raw_copy.annotations + new_annotations)

            new_entry: Any = self._update_entry_data(entry, raw_copy)
            processed_items.append(new_entry)

        log.info("Continuous preprocessing of all recordings completed successfully", VerbosityLevel.QUIET)
        return StepResult(RawPreprocessedDTO(data=processed_items))

    def _update_entry_data(self, entry: Any, new_raw_data: mne.io.Raw) -> Any:
        """
        Helper method to replace the 'data' field in various container types
        (Dataclass, NamedTuple, or generic objects).
        """
        if dataclasses.is_dataclass(entry):
            return dataclasses.replace(entry, data=new_raw_data)
        elif hasattr(entry, "_replace"):
            return entry._replace(data=new_raw_data)
        else:
            new_entry = copy.copy(entry)
            new_entry.data = new_raw_data
            return new_entry