import dataclasses
import logging
import warnings

import mne
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
        log = logging.getLogger(__name__)
        cfg: EpochPreprocessingConfig = input_dto.epoch_preprocessing_config

        log.info(f"Starting epoch preprocessing for {len(input_dto.data.data)} recordings")
        processed_recordings = []

        try:
            for i, entry in enumerate(input_dto.data.data):
                if len(entry.data) == 0:
                    log.warning(f"Entry {i} contains no epochs. Skipping.")
                    continue

                # Work on a copy of MNE Epochs
                epochs: mne.Epochs = entry.data.copy()

                # --- 1. Temporal Alignment ---
                if cfg.alignment.enabled:
                    log.info(f"Applying time shift for index {i}: {cfg.alignment.tmin_offset}s")
                    epochs.shift_time(cfg.alignment.tmin_offset, relative=True)

                # --- 2. ICA: Artifact Removal ---
                if cfg.ica.enabled:
                    log.info(f"Fitting ICA for index {i}")

                    # We suppress the baseline warning because the data is already preloaded
                    # and baseline-corrected from the previous Paradigm step.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*baseline-corrected.*")
                        ica = ICA(n_components=cfg.ica.n_components, random_state=cfg.ica.random_state, method=cfg.ica.method)
                        ica.fit(epochs)

                        # Find and exclude EOG components
                        eog_indices, _ = ica.find_bads_eog(epochs, threshold=cfg.ica.eog_threshold)
                        ica.exclude = eog_indices
                        ica.apply(epochs)

                # --- 3. AutoReject: Local Artifact Repair ---
                if cfg.autoreject.enabled:
                    log.info(f"Applying AutoReject for index {i}")

                    # Explicitly pick only EEG channels to avoid "No channels match" errors
                    # especially if CSD or other transforms were applied.
                    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")

                    if len(picks) == 0:
                        log.warning(f"No EEG channels found for AutoReject at index {i}. Skipping AR.")
                    else:
                        ar = AutoReject(n_interpolate=cfg.autoreject.n_interpolate, consensus=cfg.autoreject.consensus, cv=cfg.autoreject.cv, random_state=cfg.ica.random_state, picks=picks, verbose=False)
                        epochs, _ = ar.fit_transform(epochs, return_log=True)

                # --- 4. CSP & Data Formatting ---
                if cfg.csp.enabled:
                    log.info(f"Applying CSP and converting to ndarray for index {i}")
                    labels = epochs.events[:, -1]
                    csp = CSP(n_components=cfg.csp.n_components, reg=cfg.csp.reg, log=cfg.csp.log, norm_trace=cfg.csp.norm_trace)

                    # Transform to (n_epochs, n_csp_components)
                    signal_data = csp.fit_transform(epochs.get_data(), labels)
                    new_entry = dataclasses.replace(entry, data=signal_data)
                else:
                    new_entry = dataclasses.replace(entry, data=epochs)

                processed_recordings.append(new_entry)

            return StepResult(EpochPreprocessedDTO(data=processed_recordings))

        except Exception as e:
            log.error(f"Error in EpochPreprocessor at index {i}: {str(e)}")
            raise
