from pydantic import BaseModel, model_validator

from src.types.dto.config.logging.log_format import LogFormat
from src.types.dto.config.logging.verbosity_level import VerbosityLevel


def is_blank(txt: str | None) -> bool:
    return txt is None or txt.strip() == ""


class LoggingConfig(BaseModel):
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    console: bool = True
    file: bool = False
    file_path: str | None = None
    format: LogFormat = LogFormat.PLAIN

    @model_validator(mode="after")
    def validate_file_path(self) -> "LoggingConfig":
        if self.file and is_blank(self.file_path):
            raise ValueError("file_path must be provided when file logging is enabled")

        return self
