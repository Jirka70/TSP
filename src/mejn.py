import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

# 1. Setup Dataset
dataset = BNCI2014_001()
subject_id = 1
# Get raw data to perform "Before Trial" cleaning
data = dataset.get_data(subjects=[subject_id])
session = list(data[subject_id].keys())[0]
run = list(data[subject_id][session].keys())[0]
raw = data[subject_id][session][run]
raw.load_data()

# --- STEP 1: BEFORE TRIAL (RAW) PREPROCESSING ---
# A. High-pass filter (0.5 Hz is standard to remove DC drift)
raw.filter(l_freq=0.5, h_freq=None, fir_design="firwin")

# B. Notch filter (50Hz power line noise)
raw.notch_filter(freqs=[50.0])

# C. Automated Bad Channel Detection
# We use the standard deviation to find flat or extremely noisy channels
raw.info["bads"] = []
# For demonstration, let's use MNE's find_bad_channels_maxwell-style logic or simple variance
# Here we will plot to let you manually inspect, or use a simple threshold:
print(f"Initial Channels: {raw.ch_names}")

# --- STEP 2: USE PARADIGM (EPOCHING) ---
# We use LeftRightImagery which defines the 8-32Hz band and time window
paradigm = LeftRightImagery(fmin=8, fmax=32)
# get_data returns: X (numpy), labels, and meta.
# However, to use ICA on EPOCHS, we need the MNE Epochs object:
epochs, labels, meta = paradigm.get_data(
    dataset=dataset, subjects=[subject_id], return_epochs=True
)

# --- STEP 3: ICA FILTER ON EPOCHS ---
# ICA is used to remove artifacts like eye blinks (EOG)
ica = ICA(n_components=15, random_state=97, max_iter=800)
ica.fit(epochs)

# Visualize ICA components to identify artifacts
ica.plot_components()
# Typically, you would exclude components here, e.g., ica.exclude = [0]
# For this script, we apply it to clean the signal
epochs_cleaned = ica.apply(epochs.copy())

# --- STEP 4: VISUALIZATION ---
# A. Compare Raw vs Cleaned Epochs
print("Visualizing Cleaned Epochs...")
epochs_cleaned.plot(title="Cleaned Motor Imagery Epochs", n_epochs=5)

# B. Visualize the Power Spectral Density (PSD)
epochs_cleaned.compute_psd().plot()
plt.suptitle("PSD of Cleaned Data (8-32 Hz Focus)")

plt.show()
