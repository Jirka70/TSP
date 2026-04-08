"""
Standalone smoke-test for RiemannianModelTrainer.

No pipeline, no config files, no DI framework.

Flow:
    MOABB BNCI2014_001 (subject 1, all sessions/runs)
        -> mne.io.Raw
        -> bandpass filter + mne.Epochs
        -> EpochingDataDTO  (shuffled)
        -> TrainingInputDTO  (80/20 split)
        -> RiemannianModelTrainer.run()
        -> stratified 5-fold cross-validation report
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import mne
import numpy as np
import moabb.datasets as moabb_datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when running the script directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impl.model.riemannian_trainer import RiemannianModelTrainer
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (mirror configs/dataset/eegbci.yaml + configs/paradigm/testing.yaml)
# ---------------------------------------------------------------------------
SUBJECT_ID = 1
# None = load every session and every run the dataset provides
RUN_IDS: list[int] | None = None
EVENTS = {"left_hand": 1, "right_hand": 2}   # BNCI2014_001 event codes
FMIN, FMAX = 8.0, 35.0
TMIN, TMAX = -0.5, 4.0
BASELINE = (-0.5, 0.0)
RESAMPLE_HZ = 128.0
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Step 1 – load raw recordings from MOABB
# ---------------------------------------------------------------------------
def load_raw_runs(subject_id: int, run_ids: list[int] | None) -> list[mne.io.Raw]:
    log.info(
        "Loading BNCI2014_001 subject=%d run_ids=%s …",
        subject_id,
        run_ids if run_ids is not None else "ALL",
    )
    dataset = moabb_datasets.BNCI2014_001()
    all_data = dataset.get_data(subjects=[subject_id])

    raws: list[mne.io.Raw] = []
    sessions = all_data[subject_id]
    for session_id, runs in sessions.items():
        for run_id, raw in runs.items():
            if run_ids is None or int(run_id) in run_ids:
                log.info("  session=%s  run=%s  sfreq=%.1f Hz", session_id, run_id, raw.info["sfreq"])
                raws.append(raw)

    if not raws:
        raise RuntimeError(f"No runs found for subject {subject_id}, run_ids={run_ids}")

    return raws


# ---------------------------------------------------------------------------
# Step 2 – raw -> mne.Epochs  (same logic as ParadigmPreprocessor)
# ---------------------------------------------------------------------------
def raw_to_epochs(raw: mne.io.Raw) -> mne.Epochs:
    raw = raw.copy()
    raw.filter(l_freq=FMIN, h_freq=FMAX, fir_design="firwin", skip_by_annotation="edge")

    events_arr, event_id = mne.events_from_annotations(raw)
    event_id_filtered = {k: v for k, v in event_id.items() if k in EVENTS}

    if not event_id_filtered:
        raise RuntimeError(
            f"None of the expected events {list(EVENTS)} found in raw. "
            f"Available: {list(event_id)}"
        )

    epochs = mne.Epochs(
        raw,
        events=events_arr,
        event_id=event_id_filtered,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        reject_by_annotation=True,
        preload=True,
    )
    epochs.resample(RESAMPLE_HZ)
    return epochs


# ---------------------------------------------------------------------------
# Step 3 – mne.Epochs -> EpochingDataDTO
# ---------------------------------------------------------------------------
def epochs_to_dto(epochs: mne.Epochs) -> EpochingDataDTO:
    labels: list[int] = list(epochs.events[:, 2])
    event_names: list[str] = [
        next(k for k, v in epochs.event_id.items() if v == lbl)
        for lbl in labels
    ]

    return EpochingDataDTO(
        data=epochs,
        labels=labels,
        event_names=event_names,
        sampling_rate_hz=float(epochs.info["sfreq"]),
        n_epochs=len(epochs),
        n_channels=len(epochs.ch_names),
        n_times=len(epochs.times),
        channel_names=list(epochs.ch_names),
    )


# ---------------------------------------------------------------------------
# Step 4 – train / validation split (simple index split)
# ---------------------------------------------------------------------------
def make_dto_from_array(
    data: np.ndarray,
    labels: list[int],
    event_names: list[str],
    source: EpochingDataDTO,
) -> EpochingDataDTO:
    return EpochingDataDTO(
        data=data,
        labels=labels,
        event_names=event_names,
        sampling_rate_hz=source.sampling_rate_hz,
        n_epochs=len(labels),
        n_channels=source.n_channels,
        n_times=source.n_times,
        channel_names=source.channel_names,
    )


def split_dto(
    dto: EpochingDataDTO,
    ratio: float,
    seed: int = RANDOM_SEED,
) -> tuple[EpochingDataDTO, EpochingDataDTO]:
    raw_data: np.ndarray = (
        dto.data.get_data() if hasattr(dto.data, "get_data") else dto.data
    )

    indices = np.arange(dto.n_epochs)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split = int(dto.n_epochs * ratio)
    train_idx, val_idx = indices[:split], indices[split:]

    train = make_dto_from_array(
        raw_data[train_idx],
        [dto.labels[i] for i in train_idx],
        [dto.event_names[i] for i in train_idx],
        dto,
    )
    val = make_dto_from_array(
        raw_data[val_idx],
        [dto.labels[i] for i in val_idx],
        [dto.event_names[i] for i in val_idx],
        dto,
    )
    return train, val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def test_riemannian_trainer() -> None:
    # 1. Load all sessions/runs
    raws = load_raw_runs(SUBJECT_ID, RUN_IDS)

    # 2. Epoch each run and concatenate
    epoch_list = [raw_to_epochs(r) for r in raws]
    all_epochs = mne.concatenate_epochs(epoch_list)
    log.info("Total epochs after concatenation: %d", len(all_epochs))

    # 3. Wrap in DTO and extract numpy array up front
    full_dto = epochs_to_dto(all_epochs)
    log.info(
        "EpochingDataDTO: n_epochs=%d  n_channels=%d  n_times=%d  sfreq=%.1f",
        full_dto.n_epochs, full_dto.n_channels, full_dto.n_times, full_dto.sampling_rate_hz,
    )

    X: np.ndarray = all_epochs.get_data()          # (n_epochs, n_ch, n_times)
    y: np.ndarray = np.array(full_dto.labels)

    mock_run_ctx = MagicMock()
    mock_run_ctx.run_id = "smoke-test-001"
    mock_run_ctx.dataset_name = "BNCI2014_001"

    # ------------------------------------------------------------------
    # A) Single train/val split (shuffled)
    # ------------------------------------------------------------------
    train_dto, val_dto = split_dto(full_dto, TRAIN_RATIO)
    log.info("Single split — Train: %d  |  Val: %d", train_dto.n_epochs, val_dto.n_epochs)

    trainer = RiemannianModelTrainer()
    result: StepResult[TrainedModelDTO] = trainer.run(
        TrainingInputDTO(config=MagicMock(), train_data=train_dto, validation_data=val_dto),
        mock_run_ctx,
    )
    dto: TrainedModelDTO = result.data
    history = dto.history

    print("\n" + "=" * 60)
    print(f"  Model         : {dto.model_name}")
    print(f"  Train accuracy: {history.train_metrics['accuracy'][0]:.4f}")
    if history.validation_metrics:
        print(f"  Val accuracy  : {history.validation_metrics['accuracy'][0]:.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # B) Stratified k-fold cross-validation (no pipeline involved)
    # ------------------------------------------------------------------
    log.info("Running %d-fold stratified cross-validation …", CV_FOLDS)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_train = make_dto_from_array(
            X[train_idx], y[train_idx].tolist(),
            [full_dto.event_names[i] for i in train_idx], full_dto,
        )
        fold_val = make_dto_from_array(
            X[val_idx], y[val_idx].tolist(),
            [full_dto.event_names[i] for i in val_idx], full_dto,
        )

        fold_result = RiemannianModelTrainer().run(
            TrainingInputDTO(config=MagicMock(), train_data=fold_train, validation_data=fold_val),
            mock_run_ctx,
        )
        fold_acc = fold_result.data.history.validation_metrics["accuracy"][0]
        cv_scores.append(fold_acc)
        log.info("  Fold %d/%d — val accuracy: %.4f", fold, CV_FOLDS, fold_acc)

    cv_scores_arr = np.array(cv_scores)
    print("\n" + "=" * 60)
    print(f"  {CV_FOLDS}-fold CV results")
    print(f"  Per-fold : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean     : {cv_scores_arr.mean():.4f}")
    print(f"  Std      : {cv_scores_arr.std():.4f}")
    print(f"  Min/Max  : {cv_scores_arr.min():.4f} / {cv_scores_arr.max():.4f}")
    print("=" * 60 + "\n")

    log.info("Smoke test passed.")


if __name__ == "__main__":
    test_riemannian_trainer()
