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
    Implements a epoch_preprocessing strategy for the mne.io.Raw data structure.

    This class handles operations that require temporal stability or are
    sensitive to discontinuities. Key responsibilities include the
    identification and interpolation of defective sensors,
    global frequency filtering to remove slow drifts, and spatial
    re-referencing (e.g., CAR or CSD) to enhance the topographical specificity
    of Motor Imagery activity.
    """

    def run(self, input_dto: RawPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[RawPreprocessedDTO]:
        r"""
        Executes the continuous signal cleaning pipeline for all input runs.

        The execution sequence follows these optimal analytical steps:
        1. Identification and interpolation of defective channels.
        2. Application of a high-pass filter and Notch filter.
        3. Transformation to Current Source Density (CSD).
        4. Automatic annotation of macroscopic artifacts.
        """
        log = logging.getLogger(__name__)
        cfg: RawPreprocessingConfig = input_dto.raw_preprocessing_config
        log.info(f"Starting processing of {len(input_dto.data.data)} continuous EEG recordings")

        processed_items = []

        for i, entry in enumerate(input_dto.data.data):
            log.info(f"Processing recording index: {i}")

            # Copy continuous signal to avoid modifying the original data in the input DTO
            raw_copy: mne.io.Raw = entry.data.copy()

            # 1. Identification and interpolation of bad channel
            if raw_copy.info["bads"]:
                log.info(f"Interpolating bad channels at index {i}: {raw_copy.info['bads']}")
                raw_copy.interpolate_bads(reset_bads=True)
            else:
                log.info(f"No bad channels detected for interpolation at index {i}")

            # 2. High-pass and Notch filtration
            log.info(f"Applying filters to index {i}: HPF {cfg.high_pass_filter.l_freq} Hz, Notch {list(cfg.notch_filter.freqs)} Hz")
            raw_copy.filter(l_freq=cfg.high_pass_filter.l_freq, h_freq=None, fir_design="firwin", skip_by_annotation="edge")
            raw_copy.notch_filter(freqs=list(cfg.notch_filter.freqs), fir_design="firwin")

            # 3. Spatial transformation - Current Source Density (CSD)
            log.info(f"Computing Current Source Density for index {i}")
            try:
                raw_copy = mne.preprocessing.compute_current_source_density(raw_copy)
            except (RuntimeError, ValueError) as e:
                log.warning(f"CSD skipped at index {i}: {e}")

            # 4. Automatic annotation of large artifacts
            new_annotations = mne.preprocessing.annotate_break(
                raw_copy,
                min_break_duration=cfg.annotate_break.min_break_duration,
                t_start_after_previous=1.0,
                t_stop_before_next=1.0,
            )
            raw_copy.set_annotations(raw_copy.annotations + new_annotations)

            new_entry = None

            if dataclasses.is_dataclass(entry):
                new_entry = dataclasses.replace(entry, data=raw_copy)

            elif hasattr(entry, "_replace"):
                new_entry = entry._replace(data=raw_copy)
            else:
                new_entry = copy.copy(entry)

            processed_items.append(new_entry)

        log.info("Continuous preprocessing of all recordings completed successfully")
        log.info(processed_items)

        # Return the StepResult containing the full list of processed entries
        return StepResult(RawPreprocessedDTO(data=processed_items))
