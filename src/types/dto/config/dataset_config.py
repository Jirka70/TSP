from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: Literal["eegbci"]
    path: Path
    subject_ids: list[int]
    session_ids: list[int] | None
    run_ids: list[int] | None
    task: str