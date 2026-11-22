import mne
import numpy as np


def get_brain_channel_signal(epochs: mne.BaseEpochs) -> np.ndarray:
    """Return the signal of EEG and CSD channels only, shaped (n_epochs, n_channels, n_times).

    Recordings can contain non-brain channels: MOABB adds a stim channel that holds the class code at the cue
    and EOG channels that record eye movements. Neither may reach the model, otherwise the model reads the label
    or the eye movements instead of decoding EEG. CSD channels are kept because CSD re-referencing changes the
    channel type from EEG to CSD. Channels marked as bad are kept, as before this selection was introduced.

    Raises:
        ValueError: If the epochs contain no EEG or CSD channel.
    """
    brain_channel_indices: np.ndarray = mne.pick_types(epochs.info, eeg=True, csd=True, exclude=[])
    if len(brain_channel_indices) == 0:
        raise ValueError(f"Epochs contain no EEG or CSD channel. Channel types: {epochs.get_channel_types()}")

    return epochs.get_data(picks=brain_channel_indices, copy=False)
