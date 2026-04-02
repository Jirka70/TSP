"""
Module for the MOABB-based paradigm orchestration and data segmentation.

This stage translates continuous EEG signals into standardized, segmented
epochs. It utilizes the MOABB framework to define experimental paradigms,
applying time-delimitation, baseline correction, and resampling as part of
a unified BCI benchmark pipeline.
"""

import logging

import mne
from moabb.paradigms import MotorImagery

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.temporary_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.temporary_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.interfaces.preprocessing import IPreprocessing


class ParadigmPreprocessor(IPreprocessing):
    """
    Orchestrates the transition from Raw signal to segmented Epochs using MOABB.

    This class configures the Motor Imagery paradigm parameters, including
    frequency bands (typically 8-30 Hz for mu/beta rhythms), trial windows,
    and resampling rates. It ensures that the output maintains neurophysiological
    metadata by enforcing the return of MNE.Epochs objects.
    """

    def run(self, input_dto: EpochPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[EpochPreprocessedDTO]:
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

        paradigm_config = {
            "events": ["left_hand", "right_hand"],
            "fmin": 8.0,
            "fmax": 35.0,
            "tmin": 0.0,
            "tmax": 4.0,
            "baseline": (-0.5, 0.0),
            "resample": 128.0,
        }

        try:
            # Currently not using the paradigm - everything done manually - might change later
            paradigm = MotorImagery(**paradigm_config)
            log.info(paradigm)
            # This part of the code

            raw = input_dto.signal

            log.info(f"Band-pass filtering: {paradigm_config['fmin']}-{paradigm_config['fmax']} Hz")
            raw.filter(
                l_freq=paradigm_config["fmin"], h_freq=paradigm_config["fmax"], fir_design="firwin", skip_by_annotation="edge"
            )

            events, event_id = mne.events_from_annotations(raw)

            event_id_filtered = {k: v for k, v in event_id.items() if k in paradigm_config["events"]}

            log.info(f"Creating epochs (segmentation) with tmin={paradigm_config['tmin']}, tmax={paradigm_config['tmax']}")
            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=event_id_filtered,
                tmin=paradigm_config["tmin"],
                tmax=paradigm_config["tmax"],
                baseline=paradigm_config["baseline"],
                reject_by_annotation=True,
                preload=True,
            )

            if paradigm_config.get("resample"):
                log.info(f"Resampling epochs to {paradigm_config['resample']} Hz [cite: 89, 90]")
                epochs.resample(paradigm_config["resample"])

            log.info(f"Successfully created {len(epochs)} epochs")
            return StepResult(EpochPreprocessedDTO(signal=epochs))

        except ValueError as e:
            log.error(f"Error in paradigm event selection or frequency range: {e}")
            raise
        except Exception as e:
            log.error(f"Unexpected error during segmentation: {e}")
            raise
