import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001

# 1. Load the Dataset
# We'll take Subject 1 from the BNCI2014_001 (4-class Motor Imagery) dataset
dataset = BNCI2014_001()
subject_id = 1
data = dataset.get_data(subjects=[subject_id])

# MOABB returns data in a nested dict: {subject: {session: {run: raw_obj}}}
# We will grab the first session and first run to demonstrate
session = list(data[subject_id].keys())[0]
run = list(data[subject_id][session].keys())[0]
raw = data[subject_id][session][run]

# Ensure data is loaded into memory for in-place modification
raw.load_data()

# --- INITIAL STATE ---
print("--- Initial State ---")
print(raw.info)
# Plot the first 5 seconds of raw data
raw.plot(
    duration=5, n_channels=10, title="Initial Raw Data (Unfiltered)", scalings="auto"
)

# 2. Basic MNE Preprocessing
# A. Band-pass filtering (Motor Imagery is usually analyzed in 8-32 Hz range)
raw.filter(l_freq=8.0, h_freq=32.0, fir_design="firwin")

# B. Notch filtering to remove power line noise (e.g., 50Hz or 60Hz)
# Note: Since we band-passed up to 32Hz, 50Hz is already mostly gone,
# but this is good practice for wider filters.
raw.notch_filter(freqs=[50.0])

# C. Re-referencing (Common Average Reference is standard for MI)
raw.set_eeg_reference("average", projection=False)

# D. Resampling (to speed up computation)
raw.resample(sfreq=128)

# --- RESULT STATE ---
print("\n--- Result State ---")
print(raw.info)
# Plot the same segment after processing
raw.plot(
    duration=5,
    n_channels=10,
    title="Preprocessed Data (Filtered & Re-referenced)",
    scalings="auto",
)

plt.show()
