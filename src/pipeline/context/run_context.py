from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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

    @property
    def output_dir(self) -> Path:
        """Returns the directory where artifacts should be saved, prioritizing Hydra's output dir."""
        try:
            from hydra.core.hydra_config import HydraConfig

            path = Path(HydraConfig.get().runtime.output_dir).absolute()
        except (ValueError, KeyError, RuntimeError, ImportWarning):
            # Fallback to manual path if Hydra is not initialized
            path = Path("outputs") / f"{self.started_at.strftime('%Y-%m-%d_%H-%M-%S')}_{self.run_id[:8]}"

        path.mkdir(parents=True, exist_ok=True)
        return path
