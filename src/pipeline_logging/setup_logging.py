import logging

from src.pipeline_logging.format.Json_formatter import JsonFormatter
from src.types.dto.config.logging.log_format import LogFormat
from src.types.dto.config.logging.logging_config import LoggingConfig
from logging import Formatter


DEFAULT_LEVEL: int = logging.DEBUG
ENCODING = "utf-8"

def setup_pipeline_logging(config: LoggingConfig) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(DEFAULT_LEVEL)

    root_logger.handlers.clear()

    formatter: Formatter

    if config.format == LogFormat.JSON:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)

    if config.file:
        if config.file_path is None or config.file_path.strip() == "":
            raise ValueError("file_path must be provided when file pipeline_logging is enabled.")

        file_handler = logging.FileHandler(config.file_path, encoding=ENCODING)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(DEFAULT_LEVEL)
        root_logger.addHandler(file_handler)