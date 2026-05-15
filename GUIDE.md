# TSP Pipeline — Usage & Configuration Guide

## Quick Start

The pipeline uses [Hydra](https://hydra.cc/) for configuration management. Run from the project root:

```bash
# Activate virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Run with default config
python src/main.py
```

Hydra automatically saves the full resolved config and logs to `outputs/` after each run.

---

## Startup Example

This example runs a **training** pipeline on subjects 1–3 from an external MOABB dataset, using within-session
cross-validation, basic augmentation, and EEGNet:

**`configs/config.yaml`**

```yaml
defaults:
  - _self_
  - raw_preprocessing: default
  - paradigm: default
  - epoch_preprocessing: default
  - split: moabb_within_session
  - augmentation: basic
  - model: deep_learning/eegnet
  - metrics_aggregator: default
  - final_trainer: eegnet
  - evaluation: default
  - visualization: matplotlib
  - source: external
  - save_artifacts: default

mode: training
output_dir: ./outputs
```

**`configs/source/external.yaml`** — point to your dataset and subjects:

```yaml
backend: external
name: BNCI2014_001
path: ./data/eegbci
subject_ids:
  - 1
  - 2
  - 3
session_ids: null
run_ids: null
```

```bash
python src/main.py
```

---

## Modes

| Value        | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| `training`   | Full pipeline: preprocessing → split → augmentation → train → evaluate → save |
| `experiment` | Preprocessing only: load → raw preprocess → paradigm → epoch preprocess       |

Set in `config.yaml` or on the command line: `python src/main.py mode=experiment`

---

## Config Reference

All configs live under `configs/`. The root `config.yaml` selects one file per group using the `defaults` list.

### `configs/config.yaml` — Root config

| Key             | Default                | Options                                                                                               |
|-----------------|------------------------|-------------------------------------------------------------------------------------------------------|
| `mode`          | `training`             | `training`, `experiment`                                                                              |
| `output_dir`    | `./outputs`            | any path                                                                                              |
| `source`        | `external`             | `external`, `filesystem`                                                                              |
| `split`         | `moabb_within_session` | `moabb_within_session`, `moabb_within_subject`, `moabb_cross_session`, `moabb_cross_subject`, `basic` |
| `augmentation`  | `basic`                | `basic`, `torcheeg`, `none`                                                                           |
| `model`         | `deep_learning/eegnet` | `deep_learning/eegnet`, `machine_learning/csp_lda`, `machine_learning/riemannian_*`                   |
| `visualization` | `matplotlib`           | `matplotlib`, `plotly`                                                                                |
| `evaluation`    | `default`              | `default`, `sklearn`                                                                                  |

---

### `configs/source/` — Data source

#### `external.yaml` — MOABB dataset

```yaml
backend: external
name: BNCI2014_001       # MOABB dataset name
path: ./data/eegbci      # local cache path
subject_ids: [ 1 ]         # list of subject IDs to load
session_ids: null        # null = all sessions
run_ids: null            # null = all runs
```

#### `filesystem.yaml` — Local EDF files

```yaml
backend: filesystem
path: /path/to/edf/files
recursive: true
subject_ids: [ 1, 2, 3, 4, 5 ]
global_events_tsv_path: /path/to/task-motor-imagery_events.tsv
```

---

### `configs/raw_preprocessing/default.yaml` — Raw signal preprocessing

Each block can be toggled independently with `enabled: true/false`.

| Block                        | Default  | Key params                              |
|------------------------------|----------|-----------------------------------------|
| `resampling`                 | disabled | `sfreq: 250`                            |
| `high_pass_filter`           | enabled  | `l_freq: 1.0` Hz                        |
| `low_pass_filter`            | disabled | `h_freq: 40.0` Hz                       |
| `notch_filter`               | enabled  | `freqs: [50]` (60 for North America)    |
| `bad_channels_interpolation` | enabled  | —                                       |
| `re_referencing`             | enabled  | `method: CSD` or `AVERAGE`              |
| `ica`                        | disabled | `n_components: 0.95`, `method: infomax` |
| `annotate_break`             | enabled  | `min_break_duration: 2.5` s             |

---

### `configs/paradigm/default.yaml` — Epoching & filtering

```yaml
events:
  left_hand: 1            # trigger code for left hand
  right_hand: 2           # trigger code for right hand

filter:
  fmin: 8.0               # Alpha/Mu band start (Hz)
  fmax: 35.0              # Beta band end (Hz)

window:
  tmin: -0.5              # epoch start relative to event (s)
  tmax: 4.0               # epoch end relative to event (s)
  baseline: [ -0.5, 0.0 ]   # baseline correction window

resampling:
  enabled: true
  sfreq: 128.0            # target sample rate after epoching
```

---

### `configs/epoch_preprocessing/default.yaml` — Epoch-level preprocessing

| Block        | Default  | Key params                               |
|--------------|----------|------------------------------------------|
| `alignment`  | enabled  | `tmin_offset: -0.2` s                    |
| `ica`        | enabled  | `n_components: 15`, `eog_threshold: 3.0` |
| `autoreject` | enabled  | `n_interpolate: [1,4,8]`, `cv: 5`        |
| `csp`        | disabled | `n_components: 4`, `log: true`           |

---

### `configs/split/` — Data splitting

#### `moabb_within_session.yaml` (default)

```yaml
backend: moabb_within_session
enabled: true
pre_split_validation: false
validation_ratio: 0.2
evaluator:
  _target_: moabb.evaluations.WithinSessionSplitter
  n_folds: 2
  shuffle: true
  cv_class: sklearn.model_selection.StratifiedKFold
```

| Config file            | Strategy                         |
|------------------------|----------------------------------|
| `moabb_within_session` | CV within each session           |
| `moabb_within_subject` | CV within each subject           |
| `moabb_cross_session`  | Leave-one-session-out            |
| `moabb_cross_subject`  | Leave-one-subject-out            |
| `basic`                | Fixed train/val/test ratio split |

#### `basic.yaml`

```yaml
backend: basic
train_ratio: 0.7
validation_ratio: 0.1
test_ratio: 0.2
shuffle: true
random_seed: 42
```

---

### `configs/augmentation/` — Data augmentation

| Config     | Description                                         |
|------------|-----------------------------------------------------|
| `basic`    | Gaussian noise + time shift + channel dropout       |
| `torcheeg` | TorchEEG transforms (mask, shift, sign flip, scale) |
| `none`     | Disabled                                            |

#### `basic.yaml`

```yaml
enabled: true
backend: basic
copies_per_sample: 2       # augmented copies per original
gaussian_noise_std: 0.01
max_time_shift: 10         # samples
channel_dropout_prob: 0.0
random_seed: 42
```

---

### `configs/model/` — Model selection

#### Deep learning: `deep_learning/eegnet.yaml`

```yaml
backend: eegnet
n_classes: 2
n_channels: 26
n_times: 578
dropout: 0.25
kernel_length: 16
f1: 8
d: 2
f2: 16          # should equal f1 * d
fold_training: true
```

Training hyperparameters are in `model/training/default.yaml`:

```yaml
epochs: 10
batch_size: 32
learning_rate: 0.001
optimizer: adam
```

#### Machine learning: `machine_learning/csp_lda.yaml`

```yaml
backend: sklearn
model_name: csp_lda
fold_training: false
parameters:
  n_components: 4
```

Other ML options: `riemannian_svm`, `riemannian_lda`, `riemannian_lr`, `riemannian_mdm`, `riemannian_rf`

---

### `configs/evaluation/` — Metrics

#### `default.yaml`

```yaml
backend: default
metrics:
  - accuracy
  - f1_macro
  - precision_macro
  - recall_macro
compute_confusion_matrix: true
```

---

### `configs/visualization/` — Plotting

| Config       | Output format          |
|--------------|------------------------|
| `matplotlib` | Static PNG figures     |
| `plotly`     | Interactive HTML files |

#### `matplotlib.yaml` key options

```yaml
visualize_raw: true           # PSD of raw data
visualize_epochs: true        # ERP averages
visualize_augmentation: true
visualize_evaluation: true    # confusion matrix
save_plots: true
show_plots: true              # opens window — blocks execution
n_fft: 2048
```

---

### `configs/save_artifacts/default.yaml`

```yaml
backend: default
save_model: true
save_metrics: true
save_config: true
save_training_history: false
```
