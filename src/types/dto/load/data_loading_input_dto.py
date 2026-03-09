from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLoadingInputDTO:
    dataset_name: str
    dataset_path: Path
    subject_ids: list[int] # commented out for better understanding (not that important field)
    session_ids: list[int] | None # commented out for better understanding (not that important field)
    run_ids: list[int] | None # commented out for better understanding (not that important field)
    task: str
