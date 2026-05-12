import json
from pathlib import Path
from typing import Any


def write_json(self, file_path: Path, value: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            self._to_jsonable(value),
            file,
            indent=2,
            ensure_ascii=False,
        )