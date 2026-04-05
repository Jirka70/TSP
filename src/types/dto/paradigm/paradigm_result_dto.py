from dataclasses import dataclass

import mne


@dataclass(frozen=True)
class ParadigmResultDTO:
    """
    A data transfer object representing the output of a paradigm-specific epoch_preprocessing stage.

    This class serves as a finalized container for signals that have been
    tailored to a specific experimental or analytical paradigm. The frozen
    state guarantees that the preprocessed results remain constant.

    Attributes:
        signal (mne.Epochs): The segmented epochs produced by the paradigm stage.
    """

    signal: mne.Epochs
