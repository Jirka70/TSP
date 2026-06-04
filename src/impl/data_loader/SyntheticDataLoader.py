import numpy as np
import mne

from src.pipeline.contracts.step_result import StepResult
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.interfaces.data_loader import IDataLoader


class SyntheticDataLoader(IDataLoader):
    def run(self, config, run_ctx):
        sfreq = config.sfreq
        n_channels = config.n_channels
        n_trials = config.n_trials
        trial_spacing_s = 6.0

        duration_s = n_trials * trial_spacing_s + 6.0
        n_samples = int(duration_s * sfreq)

        rng = np.random.default_rng(config.random_seed)

        channel_names = [f"EEG{i + 1}" for i in range(n_channels)]
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=sfreq,
            ch_types="eeg",
        )

        times = np.arange(n_samples) / sfreq
        data = 0.05 * rng.standard_normal((n_channels, n_samples))

        onsets = []
        descriptions = []

        for trial_idx in range(n_trials):
            label = "left_hand" if trial_idx % 2 == 0 else "right_hand"
            onset = 1.0 + trial_idx * trial_spacing_s

            onsets.append(onset)
            descriptions.append(label)

            start = int(onset * sfreq)
            stop = int((onset + 4.0) * sfreq)

            # Make the classes slightly different so training has real signal.
            if label == "left_hand":
                data[0:2, start:stop] += 0.25 * np.sin(2 * np.pi * 10 * times[start:stop])
            else:
                data[2:4, start:stop] += 0.25 * np.sin(2 * np.pi * 18 * times[start:stop])

        raw = mne.io.RawArray(data, info)
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=[0.0] * len(onsets),
                description=descriptions,
            )
        )

        recording = RecordingDTO(
            data=raw,
            dataset_name="synthetic",
            subject_id=1,
            session_id="synthetic_session",
            run_id="synthetic_run",
            metadata={
                "sfreq": sfreq,
                "n_channels": n_channels,
                "channel_names": channel_names,
            },
        )

        return StepResult(RawDataDTO(data=[recording]))