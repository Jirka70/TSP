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
        log.info(f"Orchestrating MOABB Motor Imagery paradigm for {len(input_dto.data.data)} recordings")

        cfg: ParadigmConfig = input_dto.paradigm_preprocessing_config
        processed_items = []

        try:
            # We iterate through all data entries (runs)
            for i, entry in enumerate(input_dto.data.data):
                log.info(f"Segmenting recording index: {i}")

                # Access the Raw signal from the current entry
                raw: mne.io.Raw = entry.data

                # 1. Band-pass filtering
                # Note: We filter on the copy or the object depending on your pipeline's
                # memory strategy. Here we work with the object directly as it's a new step.
                log.info(f"Applying band-pass filter ({cfg.fmin}-{cfg.fmax} Hz) to index {i}")
                raw.filter(l_freq=cfg.fmin, h_freq=cfg.fmax, fir_design="firwin", skip_by_annotation="edge")

                # 2. Event extraction from annotations
                events, event_id = mne.events_from_annotations(raw)

                # Filter event_id to include only requested events from config
                event_id_filtered = {k: v for k, v in event_id.items() if k in cfg.events}

                if not event_id_filtered:
                    log.warning(f"No matching events found for index {i}. Skipping epoching.")
                    continue

                # 3. Epoching (Segmentation)
                log.info(f"Creating epochs for index {i} with tmin={cfg.tmin}, tmax={cfg.tmax}")
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

                # --- FIX STARTS HERE ---
                # Check if we actually have any epochs left after cleaning/rejection
                if len(epochs) == 0:
                    log.warning(f"All epochs were dropped for recording index {i}!")
                    log.warning(f"Drop log summary: {epochs.drop_log}")
                    continue
                    # --- FIX ENDS HERE ---

                # 4. Resampling (Now it's safe to call)
                if cfg.resample:
                    log.info(f"Resampling epochs at index {i} to {cfg.resample} Hz")
                    epochs.resample(cfg.resample)

                # 5. Reconstruct the entry (zbytek tvého kódu)
                # ...
                # We replace the 'raw' object (Raw) with the new 'epochs' object (Epochs)
                # Ensure your output DTO structure or wrapper can handle MNE.Epochs

                # Here we assume your entry has a field (like 'raw' or 'epochs')
                # that holds the signal wrapper. We'll update it to the new epochs.
                new_entry = None

                if dataclasses.is_dataclass(entry):
                    new_entry = dataclasses.replace(entry, data=epochs)

                elif hasattr(entry, "_replace"):
                    new_entry = entry._replace(data=epochs)
                else:
                    new_entry = copy.copy(entry)

                processed_items.append(new_entry)

            log.info(f"Successfully processed all runs. Total items: {len(processed_items)}")

            # Return the result with the full list of processed entries
            return StepResult(ParadigmResultDTO(data=processed_items))

        except ValueError as e:
            log.error(f"Error in paradigm event selection or frequency range: {e}")
            raise
        except Exception as e:
            log.error(f"Unexpected error during segmentation: {e}")
            raise
