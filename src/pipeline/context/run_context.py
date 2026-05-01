from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# ============================================================
# RUN CONTEXT
# ============================================================
# This class does not carry the domain data of an EEG step itself.
# It only carries metadata about the pipeline / experiment run.
#
# Purpose:
# - to keep commit hash, run_id, seed, etc. out of DTOs
#   that describe the domain operation itself
# - to give every pipeline step access to shared runtime context
# - to make experiments easy to log and reproduce
# ============================================================
@dataclass(frozen=True)
class RunContext:
    run_id: str
    started_at: datetime
    raw_preprocessing_backend: str
    paradigm_backend: str
    epoch_preprocessing_backend: str
    augmentation_backend: str | None
    experiment_name: str  # experiment name (e. g. "left_right_subject_01")
    pipeline_name: str  # name of the entire pipeline (e. g. "training_pipeline_<version>")
    git_commit_hash: str | None
