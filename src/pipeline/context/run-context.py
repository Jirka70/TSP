from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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
    experiment_name: str
    pipeline_name: str
    git_commit_hash: Optional[str]
    random_seed: int
    started_at: datetime
