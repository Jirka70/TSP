import logging

import mne
import numpy as np
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
    Performs high-level signal cleaning and transformation on MNE Epochs.

    This stage typically involves Independent Component Analysis (ICA) for
    eye-movement rejection or Common Spatial Patterns (CSP) to maximize
    the variance between different experimental conditions (e.g., left vs. right hand).
    """

    def run(self, input_dto: EpochPreprocessingInputDTO, run_ctx: RunContext) -> StepResult[EpochPreprocessedDTO]:
        log = logging.getLogger(__name__)
        cfg: EpochPreprocessingConfig = input_dto.epoch_preprocessing_config

        log.info(f"Starting advanced epoch preprocessing for {len(input_dto.data)} recordings")

        processed_recordings = []

        try:
            for i, entry in enumerate(input_dto.data):
                # Skip empty recordings to prevent ICA/CSP crashes
                if len(entry.data) == 0:
                    log.warning(f"Recording index {i} has no epochs. Skipping.")
                    continue

                log.info(f"Processing epochs for recording index: {i}")

                # 1. Work with a copy of the epochs
                # entry.data is currently mne.Epochs
                epochs: mne.Epochs = entry.data.copy()

                # --- 1. ICA: Artifact Rejection ---
                log.info(f"Fitting ICA for index {i}")
                ica: ICA = ICA(n_components=cfg.ica.n_components, random_state=cfg.ica.random_state, method=cfg.ica.method)

                ica.fit(epochs)
                # Find components correlating with eye movements
                eog_indices, _ = ica.find_bads_eog(epochs, threshold=cfg.ica.eog_threshold)

                log.info(f"ICA (index {i}): Excluding {len(eog_indices)} components")
                ica.exclude = eog_indices
                ica.apply(epochs)

                # --- 2. CSP: Common Spatial Patterns ---
                log.info(f"Applying Common Spatial Patterns (CSP) for index {i}")
                labels: np.ndarray = epochs.events[:, -1]

                # Oprava: v MNE se používá 'ledoit_wolf' s podtržítkem
                csp: CSP = CSP(
                    n_components=cfg.csp.n_components,
                    reg="ledoit_wolf",  # Tady byla ta zrada
                    log=cfg.csp.log,
                    norm_trace=cfg.csp.norm_trace,
                )

                # Transform epochs into CSP space (features)
                x_transformed: np.ndarray = csp.fit_transform(epochs.get_data(copy=False), labels)

                log.info(f"CSP complete for index {i}. Features shape: {x_transformed.shape}")

                # --- 3. Reconstruct the frozen RecordingDTO ---
                import dataclasses

                # We create a NEW instance of RecordingDTO.
                # We replace the 'data' attribute (which was MNE.Epochs)
                # with the new 'x_transformed' (which is np.ndarray).
                new_entry = dataclasses.replace(entry, data=x_transformed)

                processed_recordings.append(new_entry)

            log.info("Advanced epoch preprocessing completed successfully for all runs")

            # Final result DTO according to your dataclass definition
            return StepResult(EpochPreprocessedDTO(data=processed_recordings))

        except Exception as e:
            log.error(f"Failed during advanced epoch preprocessing: {e}")
            raise
