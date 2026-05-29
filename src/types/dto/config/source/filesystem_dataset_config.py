from typing import Literal
from src.types.dto.config.astageconfig import AStageConfig



class FilesystemDatasetConfig(AStageConfig):
    backend: Literal["filesystem"]
    path: str
    recursive: bool
    subject_ids: list[int] | None
    global_events_tsv_path: str | None

