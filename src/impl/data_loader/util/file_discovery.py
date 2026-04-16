from pathlib import Path
from typing import Iterable


class FileDiscovery:
    """Utility responsible only for filesystem traversal."""

    def iter_directories(self, root: Path) -> Iterable[Path]:
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                yield entry

    def iter_files(self, directory: Path, recursive: bool) -> Iterable[Path]:
        iterator = directory.rglob("*") if recursive else directory.iterdir()

        for path in sorted(iterator):
            if path.is_file():
                yield path