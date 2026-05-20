

from pydantic import BaseModel, model_validator, Field

from tsp_eeg_classification.types.dto.config.logging.log_format import LogFormat
from tsp_eeg_classification.types.dto.config.logging.logging_context_config import LoggingContextConfig
from tsp_eeg_classification.types.dto.config.logging.verbosity_level import VerbosityLevel


def is_blank(txt: str | None) -> bool:
    return txt is None or txt.strip() == ""


class LoggingConfig(BaseModel):
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    console: bool = True
    file: bool = False
    file_path: str | None = None
    format: LogFormat = LogFormat.PLAIN

    context: LoggingContextConfig = Field(default_factory=LoggingContextConfig)

    @model_validator(mode="after")
    def validate_file_path(self) -> "LoggingConfig":
        if self.file and is_blank(self.file_path):
            raise ValueError("file_path must be provided when file pipeline_logging is enabled")

        return self
