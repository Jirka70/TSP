from pydantic import BaseModel

from src.types.dto.config.logging.log_format import LogFormat
from src.types.dto.config.logging.verbosity_level import VerbosityLevel


class LoggingConfig(BaseModel):
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    console: bool = True
    file: bool = False
    file_path: str | None = None
    format: LogFormat = LogFormat.PLAIN