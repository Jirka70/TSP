from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoadingInputDTO:
    dataset_name: str
    dataset_path: str
    subject_ids: list[int]
    session_ids: list[int] | None
    run_ids: list[int] | None
    task: str