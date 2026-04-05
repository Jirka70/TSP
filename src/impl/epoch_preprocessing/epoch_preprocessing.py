import logging

from mne.decoding import CSP
from mne.preprocessing import ICA

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
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
        """
        Applies advanced cleaning and spatial filters to the epoched signal.

        The process focuses on:
        1. **Artifact Cleaning**: Using ICA to decompose the signal and remove
           non-brain components (EOG, ECG).
        2. **Spatial Transformation**: Applying CSP to enhance the signal-to-noise
           ratio of motor-related rhythms.
        3. **Final Scaling**: Ensuring the data is normalized for the estimator.

        Args:
            input_dto: DTO containing segmented MNE.Epochs from the paradigm stage.
            run_ctx: Execution context providing configuration and parameters.

        Returns:
            StepResult containing the refined and cleaned Epochs.
        """
        log = logging.getLogger(__name__)
        epochs = input_dto.signal.signal
        cfg = input_dto.epoch_preprocessing_config

        log.info(f"Starting advanced epoch epoch_preprocessing for {len(epochs)} epochs")

        try:
            # --- 1. ICA: Artifact Rejection ---
            log.info("Fitting ICA to decompose signal components.")
            ica = ICA(n_components=cfg.ica.n_components, random_state=cfg.ica.random_state, method=cfg.ica.method)
            ica.fit(epochs)

            # Automatically identify EOG-related components based on correlation with EOG channels
            eog_indices, scores = ica.find_bads_eog(epochs, threshold=cfg.ica.eog_threshold)
            log.info(f"ICA: Identified {len(eog_indices)} artifact components to exclude: {eog_indices}")

            ica.exclude = eog_indices
            ica.apply(epochs)

            # --- 2. CSP: Common Spatial Patterns ---
            # CSP labels
            log.info("Applying Common Spatial Patterns (CSP) for spatial filtering.")
            labels = epochs.events[:, -1]

            # CSP
            csp = CSP(n_components=cfg.csp.n_components, reg=cfg.csp.reg, log=cfg.csp.log, norm_trace=cfg.csp.norm_trace)

            # get_data() returns numpy array
            x_transformed = csp.fit_transform(epochs.get_data(copy=False), labels)

            log.info(f"CSP transformation complete. New data shape: {x_transformed.shape}")

            # Signal is numpy.ndarray (features)
            log.info("Epoch epoch_preprocessing completed successfully.")
            return StepResult(EpochPreprocessedDTO(signal=x_transformed))

        except Exception as e:
            log.error(f"Failed during advanced epoch epoch_preprocessing: {e}")
            raise
