import logging
import sys
from pathlib import Path

from tsp_eeg_classification.pipeline_logging.filter.PipelineOrWarningFilter import PipelineOrWarningFilter
from tsp_eeg_classification.pipeline_logging.format.json_formatter import JsonFormatter
from tsp_eeg_classification.types.dto.config.logging.log_format import LogFormat
from tsp_eeg_classification.types.dto.config.logging.logging_config import LoggingConfig
from logging import Formatter


DEBUG_LEVEL = logging.DEBUG
ENCODING = "utf-8"


def setup_bootstrap_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)


def setup_pipeline_logging(config: LoggingConfig) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(DEBUG_LEVEL)

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
    pipeline_logger.propagate = True

    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(DEBUG_LEVEL)
        console_handler.addFilter(PipelineOrWarningFilter())
        root_logger.addHandler(console_handler)

    if config.file:
        if config.file_path is None or config.file_path.strip() == "":
            raise ValueError("file_path must be provided when file pipeline_logging is enabled.")

        log_file_path = Path(config.file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, encoding=ENCODING)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(PipelineOrWarningFilter())
        file_handler.setLevel(DEBUG_LEVEL)
        root_logger.addHandler(file_handler)
