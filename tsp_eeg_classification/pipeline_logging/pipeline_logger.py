import logging
from typing import Any

from tsp_eeg_classification.types.dto.config.logging.logging_config import LoggingConfig
from tsp_eeg_classification.types.dto.config.logging.verbosity_level import VerbosityLevel, VERBOSITY_ORDER

PipelineLoggerContext = dict[str, Any] | None


def _validate_message(message: str | None) -> None:
    if message is None:
        raise ValueError("Message cannot be defined as None")


def _format_context_value(value: Any) -> str:
    text = str(value)
    if any(char.isspace() for char in text):
        return f'"{text}"'

    return text


class PipelineLogger:
    def __init__(self,
                 logger: logging.Logger,
                 logging_config: LoggingConfig,
                 verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
                 ctx: PipelineLoggerContext = None) -> None:
        self._context = dict(ctx) if ctx is not None else {}
        self._logging_config = logging_config
        self._logger = logger
        self._verbosity = verbosity

    def info(self, message: str, verbosity: VerbosityLevel = VerbosityLevel.NORMAL) -> None:
        _validate_message(message)

        if self._should_log(verbosity):
            self._logger.info(self._format(message))

    def debug(self, message: str, verbosity: VerbosityLevel = VerbosityLevel.TRACE) -> None:
        _validate_message(message)

        if self._should_log(verbosity):
            self._logger.debug(self._format(message))

    def for_step(self, step: str) -> "PipelineLogger":
        return PipelineLogger(
            logger=self._logger,
            logging_config=self._logging_config,
            verbosity=self._verbosity,
            ctx={**self._context, "step": step},
        )

    def warning(self, message: str) -> None:
        _validate_message(message)

        self._logger.warning(self._format(message))

    def error(self, message: str) -> None:
        _validate_message(message)

        self._logger.error(self._format(message))

    def exception(self, message: str) -> None:
        _validate_message(message)

        self._logger.exception(self._format(message))

    def _format(self, message: str) -> str:
        UNKNOWN = "unknown"
        if message is None:
            raise ValueError("Message cannot be defined as None")

        if not self._context:
            return message

        context_config = self._logging_config.context
        parts: list[str] = []

        if context_config.run_id:
            parts.append(f"run_id={_format_context_value(self._context.get('run_id', UNKNOWN))}")

        if context_config.git_commit_hash:
            parts.append(f"git={_format_context_value(self._context.get('git_commit_hash', UNKNOWN))}")

        if context_config.pipeline_name:
            parts.append(f"pipeline={_format_context_value(self._context.get('pipeline_name', UNKNOWN))}")

        if context_config.step and self._context.get("step") is not None:
            parts.append(f"step={_format_context_value(self._context['step'])}")

        context = " ".join(parts)
        return f"[{context}] - {message}" if context else message

    def _should_log(self, message_verbosity: VerbosityLevel) -> bool:
        return VERBOSITY_ORDER[message_verbosity] <= VERBOSITY_ORDER[self._verbosity]
