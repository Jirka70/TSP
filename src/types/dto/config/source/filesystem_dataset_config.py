from src.types.dto.config.astageconfig import AStageConfig
from typing import Literal



class FilesystemDatasetConfig(AStageConfig):
    backend: Literal["filesystem"]
    path: str
    recursive: bool
    subject_ids: list[int]
    sessions_ids: list[int]

