from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    subject_ids: list[int]
    session_ids: list[int] | None
    run_ids: list[int] | None
    task: str