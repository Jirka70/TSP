import copy
import dataclasses
import logging

import mne
import mne.io

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
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
        """
        Executes the continuous signal cleaning pipeline based on the provided configuration.
        """
        log = logging.getLogger(__name__)
        cfg: RawPreprocessingConfig = input_dto.raw_preprocessing_config

        # Note: General config parameters like cfg.backend are preserved in the DTO
        log.warning("New RAW preprocessing")
        log.info(f"Starting processing with backend: {getattr(cfg, 'backend', 'default')}")
        log.info(f"Processing {len(input_dto.data.data)} continuous EEG recordings")

        processed_items = []

        for i, entry in enumerate(input_dto.data.data):
            log.info(f"Processing recording index: {i}")

            # Copy continuous signal to avoid modifying the original data in the input DTO
            raw_copy: mne.io.Raw = entry.data.copy()

            # --- 1. Resampling ---
            if getattr(cfg.resampling, "enabled", False):
                log.info(f"Resampling signal to {cfg.resampling.sfreq} Hz")
                raw_copy.resample(sfreq=cfg.resampling.sfreq)

            # --- 2. Bad Channels Identification and Interpolation ---
            if getattr(cfg.bad_channels_interpolation, "enabled", False):
                if raw_copy.info["bads"]:
                    log.info(f"Interpolating bad channels at index {i}: {raw_copy.info['bads']}")
                    raw_copy.interpolate_bads(reset_bads=True)
                else:
                    log.info(f"No bad channels detected for interpolation at index {i}")

            # --- 3. Frequency Filtering (High-pass, Low-pass, Notch) ---
            # We combine HPF and LPF into one call for better efficiency
            hpf_enabled = getattr(cfg.high_pass_filter, "enabled", False)
            lpf_enabled = getattr(cfg.low_pass_filter, "enabled", False)

            if hpf_enabled or lpf_enabled:
                l_freq = cfg.high_pass_filter.l_freq if hpf_enabled else None
                h_freq = cfg.low_pass_filter.h_freq if lpf_enabled else None
                log.info(f"Applying filter: HPF={l_freq} Hz, LPF={h_freq} Hz")
                raw_copy.filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    fir_design="firwin",
                    skip_by_annotation="edge"
                )

            if getattr(cfg.notch_filter, "enabled", False):
                log.info(f"Applying Notch filter: {list(cfg.notch_filter.freqs)} Hz")
                raw_copy.notch_filter(freqs=list(cfg.notch_filter.freqs), fir_design="firwin")

            # --- 4. ICA (Artifact Rejection) ---
            if getattr(cfg.ica, "enabled", False):
                log.info(f"Running ICA decomposition (method: {cfg.ica.method})")
                ica = mne.preprocessing.ICA(
                    n_components=cfg.ica.n_components,
                    method=cfg.ica.method,
                    random_state=42
                )
                # ICA is typically fitted on data without slow drifts (HPF > 1Hz recommended)
                ica.fit(raw_copy)
                log.info(f"Applying ICA to remove artifact components")
                ica.apply(raw_copy)

            # --- 5. Spatial Transformation / Re-referencing ---
            if getattr(cfg.re_referencing, "enabled", False):
                ref_method = cfg.re_referencing.method.upper()

                if ref_method == "CSD":
                    log.info("Computing Current Source Density (CSD)")
                    try:
                        raw_copy = mne.preprocessing.compute_current_source_density(raw_copy)
                    except (RuntimeError, ValueError) as e:
                        log.warning(f"CSD skipped at index {i}: {e}")

                elif ref_method == "AVERAGE":
                    log.info("Applying Common Average Reference (CAR)")
                    raw_copy.set_eeg_reference(ref_channels="average")

            # --- 6. Automatic Annotation of Artifacts/Breaks ---
            if getattr(cfg.annotate_break, "enabled", False):
                log.info("Generating break annotations")
                new_annotations = mne.preprocessing.annotate_break(
                    raw_copy,
                    min_break_duration=cfg.annotate_break.min_break_duration,
                    t_start_after_previous=1.0,
                    t_stop_before_next=1.0,
                )
                raw_copy.set_annotations(raw_copy.annotations + new_annotations)

            # Update entry with processed data
            new_entry = self._update_entry_data(entry, raw_copy)
            processed_items.append(new_entry)

        log.info("Continuous preprocessing of all recordings completed successfully")
        return StepResult(RawPreprocessedDTO(data=processed_items))

    def _update_entry_data(self, entry, new_raw_data):
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