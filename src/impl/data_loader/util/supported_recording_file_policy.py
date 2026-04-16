from pathlib import Path


class SupportedRecordingFilePolicy:
    _SUPPORTED_SUFFIXES = {
        ".edf",
        ".bdf",
        ".gdf",
        ".vhdr",
        ".set",
        ".fif",
    }

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in cls._SUPPORTED_SUFFIXES