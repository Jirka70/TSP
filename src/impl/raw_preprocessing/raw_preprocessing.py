import logging
from pathlib import Path

import mne
from omegaconf import OmegaConf

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDto
from src.types.interfaces.raw_preprocessing import IRawPreprocessing

_CONFIG_PATH = Path(__file__).parent / "testing.yaml"


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

    def run(self, input_dto: RawPreprocessingInputDto, run_ctx: RunContext) -> StepResult[RawPreprocessedDTO]:
        r"""
        Executes the continuous signal cleaning pipeline.

        The execution sequence follows these optimal analytical steps:
        1. Identification and interpolation of defective channels using
           spherical splines to preserve topological structure.
        2. Application of a high-pass filter (typically $\ge 1.0$ Hz) to ensure
           signal stationarity, which is a prerequisite for future Independent
           Component Analysis (ICA).
        3. Removal of power line interference via a Notch filter at 50/60 Hz.
        4. Transformation to Current Source Density (CSD) to reduce volume
           conduction and better localize signals to the sensorimotor cortex.
        5. Algorithmic or manual annotation of macroscopic artifacts (e.g.,
           movement) using 'BAD_' labels for automatic rejection during
           the paradigm process.

        Args:
            input_dto: DTO containing the raw EEG signal and hardware metadata.
            run_ctx: The context of the current pipeline execution.

        Returns:
            A StepResult containing the preprocessed Raw object, optimized
            for entry into the MOABB paradigm.
        """
        log = logging.getLogger(__name__)
        cfg = OmegaConf.load(_CONFIG_PATH)
        log.info("Starting processing of continuous EEG data (MNE.Raw)")

        # Copy continuous signal to avoid modifying the original data
        raw_data_copy = input_dto.signal.copy()

        # Identification and interpolation of bad channels
        if raw_data_copy.info["bads"]:
            log.info(f"Interpolating bad channels: {raw_data_copy.info['bads']}")
            raw_data_copy.interpolate_bads(reset_bads=True)
        else:
            log.info("No bad channels detected for interpolation")

        # High-pass and Notch filtration
        log.info(f"Applying frequency filters: HPF {cfg.high_pass_filter.l_freq} Hz and Notch {list(cfg.notch_filter.freqs)} Hz")

        raw_data_copy.filter(l_freq=cfg.high_pass_filter.l_freq, h_freq=None, fir_design="firwin", skip_by_annotation="edge")
        raw_data_copy.notch_filter(freqs=list(cfg.notch_filter.freqs), fir_design="firwin")

        # Spatial transformation - Current Source Density (CSD)
        log.info("Computing Current Source Density (Surface Laplacian)")
        try:
            raw_data_copy = mne.preprocessing.compute_current_source_density(raw_data_copy)
        except RuntimeError as e:
            log.warning(f"CSD skipped: Montage/coordinates are missing in Raw.info. Error: {e}")
        except ValueError as e:
            log.warning(f"CSD skipped: Invalid channel types or insufficient sensors. Error: {e}")

        # Automatic annotation of large artifacts
        annotations, _ = mne.preprocessing.annotate_break(raw_data_copy, min_break_duration=cfg.annotate_break.min_break_duration)

        log.info("Continuous epoch_preprocessing completed successfully")

        return StepResult(RawPreprocessedDTO(signal=raw_data_copy))
