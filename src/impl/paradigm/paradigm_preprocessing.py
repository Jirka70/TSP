import logging

import mne
import mne.io
import numpy as np
from moabb.paradigms import MotorImagery

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.paradigm_config import ParadigmConfig
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from src.types.interfaces.paradigm import IParadigm


class ParadigmPreprocessor(IParadigm):
    """
    Orchestrates the transition from Raw signal to segmented Epochs using MOABB.

    This class configures the Motor Imagery paradigm parameters, including
    frequency bands (typically 8-30 Hz for mu/beta rhythms), trial windows,
    and resampling rates. It ensures that the output maintains neurophysiological
    metadata by enforcing the return of MNE.Epochs objects.
    """

    def run(self, input_dto: ParadigmInputDTO, run_ctx: RunContext) -> StepResult[ParadigmResultDTO]:
        r"""
        Executes the segmentation and standardizing pipeline via MOABB.

        The implementation focuses on these critical parameters from the analysis:
        1. **Events**: Selecting specific trial labels (e.g., 'left_hand', 'right_hand')[cite: 88].
        2. **Time Window**: Defining tmin and tmax (e.g., 0.0 to 4.0s) to isolate
           the cognitive task.
        3. **Frequency Specification**: Applying band-pass filters (fmin=8, fmax=30)
           immediately before segmentation.
        4. **Baseline Correction**: Subtracting pre-stimulus mean voltage to
           eliminate slow drifts.
        5. **Resampling**: Downsampling (e.g., to 128 Hz) to accelerate
           subsequent deep learning training.

        Args:
            input_dto: DTO containing the pre-cleaned continuous Raw signal.
            run_ctx: The context of the current pipeline execution.

        Returns:
            StepResult containing segmented MNE.Epochs for advanced refinement.
        """
        log = logging.getLogger(__name__)
        log.info("Orchestrating MOABB Motor Imagery paradigm")

        cfg: ParadigmConfig = input_dto.paradigm_preprocessing_config

        try:
            # Currently not using the paradigm - everything done manually - might change later
            paradigm: MotorImagery = MotorImagery(
                events=list(cfg.events),
                fmin=cfg.fmin,
                fmax=cfg.fmax,
                tmin=cfg.tmin,
                tmax=cfg.tmax,
                resample=cfg.resample,
            )
            log.info(paradigm)
            # This part of the code

            raw: mne.io.Raw = input_dto.signal.signal

            log.info(f"Band-pass filtering: {cfg.fmin}-{cfg.fmax} Hz")
            raw.filter(l_freq=cfg.fmin, h_freq=cfg.fmax, fir_design="firwin", skip_by_annotation="edge")

            events: np.ndarray
            event_id: dict[str, int]
            events, event_id = mne.events_from_annotations(raw)

            event_id_filtered: dict[str, int] = {k: v for k, v in event_id.items() if k in cfg.events}

            log.info(f"Creating epochs (segmentation) with tmin={cfg.tmin}, tmax={cfg.tmax}")
            epochs: mne.Epochs = mne.Epochs(
                raw,
                events=events,
                event_id=event_id_filtered,
                tmin=cfg.tmin,
                tmax=cfg.tmax,
                baseline=tuple(cfg.baseline) if cfg.baseline is not None else None,
                reject_by_annotation=cfg.reject_by_annotation,
                preload=cfg.preload,
            )

            if cfg.resample:
                log.info(f"Resampling epochs to {cfg.resample} Hz")
                epochs.resample(cfg.resample)

            log.info(f"Successfully created {len(epochs)} epochs")
            return StepResult(ParadigmResultDTO(signal=epochs))

        except ValueError as e:
            log.error(f"Error in paradigm event selection or frequency range: {e}")
            raise
        except Exception as e:
            log.error(f"Unexpected error during segmentation: {e}")
            raise
