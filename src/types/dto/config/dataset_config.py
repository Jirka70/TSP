from pathlib import Path
from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class DatasetConfig(AStageConfig):
    backend: Literal["eegbci"]
    name: str
    path: Path
    subject_ids: list[int]
    session_ids: list[int] | None
    run_ids: list[int] | None
    paradigm: str
