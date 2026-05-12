import json
from pathlib import Path
from typing import Any

from src.impl.model.util.serialize.to_jsonable import to_jsonable

MODE = "w"
ENDODING = "utf-8"


def write_json(file_path: Path, value: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(MODE, encoding=ENDODING) as file:
        json.dump(
            to_jsonable(value),
            file,
            indent=2,
            ensure_ascii=False,
        )