from pathlib import Path


class SupportedRecordingFilePolicy:
    """
    Utility class for checking whether a file is a supported EEG recording format.
    """

    _SUPPORTED_SUFFIXES = {
        ".edf",
        ".bdf",
        ".gdf",
        ".vhdr",
        ".set",
        ".fif",
    }

    @staticmethod
    def is_supported(file_path: Path) -> bool:
        return file_path.suffix.lower() in SupportedRecordingFilePolicy._SUPPORTED_SUFFIXES