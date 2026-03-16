from pathlib import Path
from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class DatasetConfig(AStageConfig):
    _target_class = "impl.data_loader.dummy_data_loader.DummyLoader"

    name: Literal["eegbci"]
    path: Path
    subject_ids: list[int]
    session_ids: list[int] | None
    run_ids: list[int] | None
    task: str
