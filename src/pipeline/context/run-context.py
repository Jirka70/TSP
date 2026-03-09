from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RunContext:
    run_id: str
    experiment_name: str
    pipeline_name: str
    git_commit_hash: Optional[str]
    random_seed: int
    started_at: datetime