import logging


class PipelineOrWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        is_pipeline = record.name.startswith("pipeline")
        is_warning_or_above = record.levelno >= logging.WARNING

        return is_pipeline or is_warning_or_above