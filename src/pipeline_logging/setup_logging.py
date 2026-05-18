import logging
import sys
from pathlib import Path

from src.pipeline_logging.format.Json_formatter import JsonFormatter
from src.types.dto.config.logging.log_format import LogFormat
from src.types.dto.config.logging.logging_config import LoggingConfig
from logging import Formatter


INFO_LEVEL: int = logging.INFO
DEBUG_LEVEL = logging.DEBUG
WARNING_LEVEL = logging.WARNING
ENCODING = "utf-8"

def setup_pipeline_logging(config: LoggingConfig) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(INFO_LEVEL)

    root_logger.handlers.clear()

    formatter: Formatter

    if config.format == LogFormat.JSON:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.handlers.clear()
    pipeline_logger.setLevel(DEBUG_LEVEL)
    pipeline_logger.propagate = False

    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(DEBUG_LEVEL)
        pipeline_logger.addHandler(console_handler)

        root_console_handler = logging.StreamHandler(sys.stdout)
        root_console_handler.setFormatter(formatter)
        root_console_handler.setLevel(WARNING_LEVEL)
        root_logger.addHandler(root_console_handler)

    if config.file:
        if config.file_path is None or config.file_path.strip() == "":
            raise ValueError("file_path must be provided when file pipeline_logging is enabled.")

        log_file_path = Path(config.file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, encoding=ENCODING)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(DEBUG_LEVEL)
        pipeline_logger.addHandler(file_handler)

        root_file_handler = logging.FileHandler(log_file_path, encoding=ENCODING)
        root_file_handler.setFormatter(formatter)
        root_file_handler.setLevel(WARNING_LEVEL)
        root_logger.addHandler(root_file_handler)
