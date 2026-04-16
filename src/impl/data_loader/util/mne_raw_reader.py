from pathlib import Path
from mne.io import read_raw

class MneRawReader:
    """Thin wrapper around MNE raw loading."""

    @staticmethod
    def read(self, file_path: Path) -> Any:
        try:
            return read_raw(file_path, preload=True, verbose="ERROR")
        except Exception as exc:
            raise DatasetLoadingError(
                f"Failed to load recording file: '{file_path}'."
            ) from exc