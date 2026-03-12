import secrets
import subprocess
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.experiment_config import ExperimentConfig


class RunContextFactory:
    @staticmethod
    def _resolve_git_commit_hash() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def create(
        self,
        config: ExperimentConfig,
        experiment_name: str,
        pipeline_name: str,
    ) -> RunContext:

        run_id = str(uuid4())
        started_at = datetime.utcnow()

        git_commit_hash = self._resolve_git_commit_hash()

        augmentation_backend = None
        if config.augmentation.enabled:
            augmentation_backend = config.augmentation.backend

        return RunContext(
            run_id=run_id,
            started_at=started_at,
            dataset_name=config.dataset.name,
            preprocessing_backend=config.preprocessing.backend,
            augmentation_backend=augmentation_backend,
            experiment_name=experiment_name,
            pipeline_name=pipeline_name,
            git_commit_hash=git_commit_hash,
        )
