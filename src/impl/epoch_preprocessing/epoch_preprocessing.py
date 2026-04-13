import dataclasses
import logging

import mne
from mne.preprocessing import ICA

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.epoch_preprocessing_config import EpochPreprocessingConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing


class EpochPreprocessor(IEpochPreprocessing):
    """
    Performs advanced signal cleaning on MNE Epochs.

    This stage focuses on artifact removal (e.g., eye blinks) using ICA,
    while maintaining the mne.Epochs data structure for further processing.
    """

    def run(self, input_dto: EpochPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[EpochPreprocessedDTO]:
        log = logging.getLogger(__name__)
        cfg: EpochPreprocessingConfig = input_dto.epoch_preprocessing_config

        log.info(f"Starting advanced epoch preprocessing for {len(input_dto.data.data)} recordings")

        processed_recordings = []

        try:
            for i, entry in enumerate(input_dto.data.data):
                # Skip empty recordings to prevent ICA crashes
                if len(entry.data) == 0:
                    log.warning(f"Recording index {i} has no epochs. Skipping.")
                    continue

                log.info(f"Processing epochs for recording index: {i}")

                # 1. Work with a copy to maintain immutability of input data
                # entry.data is currently mne.Epochs
                epochs: mne.Epochs = entry.data.copy()

                # --- ICA: Artifact Rejection ---
                log.info(f"Fitting ICA for index {i}")
                ica: ICA = ICA(n_components=cfg.ica.n_components, random_state=cfg.ica.random_state, method=cfg.ica.method)

                # Fit ICA on the epoched data
                ica.fit(epochs)

                # Automatically find components correlating with eye movements (EOG)
                eog_indices, _ = ica.find_bads_eog(epochs, threshold=cfg.ica.eog_threshold)

                log.info(f"ICA (index {i}): Excluding {len(eog_indices)} components")
                ica.exclude = eog_indices

                # Apply the ICA cleaning - the output remains an mne.Epochs object
                ica.apply(epochs)

                # --- Reconstruct the RecordingDTO ---
                # CRITICAL: We pass the cleaned mne.Epochs object, NOT a numpy array.
                # This preserves metadata and allows for further MNE-based steps.
                new_entry = dataclasses.replace(entry, data=epochs)
                processed_recordings.append(new_entry)

            log.info("Advanced epoch preprocessing completed. All results kept as mne.Epochs structure.")

            # Return the StepResult with preserved structure
            return StepResult(EpochPreprocessedDTO(data=processed_recordings))

            # TODO: Maybe add CSP preprocessing in future.

        except Exception as e:
            log.error(f"Failed during advanced epoch preprocessing: {e}")
            raise
