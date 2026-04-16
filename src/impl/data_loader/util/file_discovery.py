from pathlib import Path
from typing import Iterable


class FileDiscovery:
    """Utility responsible only for filesystem traversal."""

    @staticmethod
    def iter_directories(root: Path) -> Iterable[Path]:
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                yield entry

    @staticmethod
    def iter_files(directory: Path, recursive: bool) -> Iterable[Path]:
        iterator = directory.rglob("*") if recursive else directory.iterdir()

        for path in sorted(iterator):
            if path.is_file():
                yield path