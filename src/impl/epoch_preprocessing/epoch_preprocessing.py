import dataclasses
import logging
import warnings
from typing import Any, List, Tuple

import mne
import numpy as np
from autoreject import AutoReject
from mne.decoding import CSP
from mne.preprocessing import ICA

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.epoch_preprocessing_config import EpochPreprocessingConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing


class EpochPreprocessor(IEpochPreprocessing):
    """
    Advanced epoch processing pipeline.
    Ensures output matches EpochPreprocessedDTO structure.
    """

    def run(self, input_dto: EpochPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[EpochPreprocessedDTO]:
        """
            Executes sequential preprocessing steps on epoched neural data.

            This method processes a collection of MNE Epochs recordings by applying a
            configurable multi-stage pipeline:
            1. Temporal Alignment: Shifts epoch time markers.
            2. Independent Component Analysis (ICA): Isolates and removes EOG artifacts.
            3. AutoReject: Automatically detects, repairs, or drops bad local data segments.
            4. Common Spatial Patterns (CSP): Learns spatial filters and transforms
               the data into feature arrays (if enabled).

            Args:
                input_dto (EpochPreprocessingInputDTO): Input object containing the
                    MNE Epochs data and the pipeline preprocessing configurations.
                run_ctx (RunContext): Context keeping track of the current pipeline execution.

            Returns:
                StepResult[EpochPreprocessedDTO]: A step result wrapping the processed
                    recordings. Depending on `cfg.csp.enabled`, the inner `.data` fields
                    will contain either transformed NumPy ndarrays or cleaned MNE Epochs objects.

            Raises:
                Exception: Re-raises any exception caught during processing, logging the
                    exact recording index where the pipeline failed.
        """
        log: logging.Logger = logging.getLogger(__name__)
        config: EpochPreprocessingConfig = input_dto.epoch_preprocessing_config

        log.info(f"Starting epoch preprocessing for {len(input_dto.data.data)} recordings")
        processed_recordings: List[Any] = []  # Replace Any with your specific Recording Entry DTO type if available

        # Declare i outside the try block so it is safely scoped for the except block
        i: int = 0

        try:
            for i, entry in enumerate(input_dto.data.data):
                if len(entry.data) == 0:
                    log.warning(f"Entry {i} contains no epochs. Skipping.")
                    continue

                # Work on a copy of MNE Epochs
                epochs: mne.Epochs = entry.data.copy()

                # --- 1. Temporal Alignment ---
                if config.alignment.enabled:
                    log.info(f"Applying time shift for index {i}: {config.alignment.tmin_offset}s")
                    epochs.shift_time(config.alignment.tmin_offset, relative=True)

                # --- 2. ICA: Artifact Removal ---
                if config.ica.enabled:
                    log.info(f"Fitting ICA for index {i}")

                    # We suppress the baseline warning because the data is already preloaded
                    # and baseline-corrected from the previous Paradigm step.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*baseline-corrected.*")
                        ica: ICA = ICA(
                            n_components=config.ica.n_components,
                            random_state=config.ica.random_state,
                            method=config.ica.method
                        )
                        ica.fit(epochs)

                        electrooculography_indices: List[int]
                        electrooculography_indices, _ = ica.find_bads_eog(epochs, threshold=config.ica.eog_threshold)
                        ica.exclude = electrooculography_indices
                        ica.apply(epochs)

                # --- 3. AutoReject: Local Artifact Repair ---
                if config.autoreject.enabled:
                    log.info(f"Applying AutoReject for index {i}")
                    picks: np.ndarray = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")

                    if len(picks) == 0:
                        log.warning(f"No EEG channels found for AutoReject at index {i}. Skipping AR.")
                    else:
                        auto_reject: AutoReject = AutoReject(
                            n_interpolate=config.autoreject.n_interpolate,
                            consensus=config.autoreject.consensus,
                            cv=config.autoreject.cv,
                            random_state=config.ica.random_state,
                            picks=picks,
                            verbose=False
                        )
                        epochs, _ = auto_reject.fit_transform(epochs, return_log=True)

                # --- 4. CSP & Data Formatting ---
                if config.csp.enabled:
                    log.info(f"Applying CSP and converting to ndarray for index {i}")
                    labels: np.ndarray = epochs.events[:, -1]
                    csp: CSP = CSP(
                        n_components=config.csp.n_components,
                        reg=config.csp.reg,
                        log=config.csp.log,
                        norm_trace=config.csp.norm_trace
                    )

                    # Transform to (n_epochs, n_csp_components)
                    signal_data: np.ndarray = csp.fit_transform(epochs.get_data(), labels)
                    new_entry: Any = dataclasses.replace(entry, data=signal_data)
                else:
                    new_entry = dataclasses.replace(entry, data=epochs)

                processed_recordings.append(new_entry)

            return StepResult(EpochPreprocessedDTO(data=processed_recordings))

        except Exception as e:
            log.error(f"Error in EpochPreprocessor at index {i}: {str(e)}")
            raise