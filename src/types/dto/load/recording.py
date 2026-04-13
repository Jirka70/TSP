from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecordingDTO:
    data: Any
    dataset_name: str
    subject_id: int | str
    session_id: str | None
    run_id: str | int | None
    metadata: dict[str, Any] = field(default_factory=dict)
