from pathlib import Path

from mne.io import BaseRaw, read_raw

from src.impl.data_loader.error.dataset_loading_error import DatasetLoadingError



class MneRawReader:
    """Thin wrapper around MNE raw loading."""

    @staticmethod
    def read(file_path: Path) -> BaseRaw:
        try:
            return read_raw(file_path, preload=True, verbose="ERROR")
        except Exception as exc:
            raise DatasetLoadingError(
                f"Failed to load recording file: '{file_path}'."
            ) from exc