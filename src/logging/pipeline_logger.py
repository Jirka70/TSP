import logging
from typing import Any

from src.types.dto.config.logging.verbosity_level import VerbosityLevel, VERBOSITY_ORDER

type PipelineLoggerContext = dict[str, Any] | None


class PipelineLogger:
    def __init__(self,
                 logger: logging.Logger,
                 verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
                 ctx: PipelineLoggerContext = None) -> None:
        self._context = dict(ctx) if ctx is not None else {}
        self._logger = logger
        self._verbosity = verbosity
        self._context = ctx or {}

    def info(self, message: str, verbosity: VerbosityLevel = VerbosityLevel.NORMAL) -> None:
        if message is None:
            raise ValueError("Message cannot be defined as None")

        if self._should_log(verbosity):
            self._logger.info(self._format(message))

    def debug(self, message: str, verbosity: VerbosityLevel = VerbosityLevel.NORMAL) -> None:
        if message is None:
            raise ValueError("Message cannot be defined as None")

        if self._should_log(verbosity):
            self._logger.debug(self._format(message))

    def for_step(self, step: str) -> "PipelineLogger":
        return PipelineLogger(
            logger=self._logger,
            verbosity=self._verbosity,
            ctx={**self._context, "step": step},
        )

    def warning(self, message: str) -> None:
        if message is None:
            raise ValueError("Message cannot be defined as None")

        self._logger.warning(self._format(message))

    def error(self, message: str) -> None:
        if message is None:
            raise ValueError("Message cannot be defined as None")

        self._logger.error(self._format(message))

    def exception(self, message: str) -> None:
        if message is None:
            raise ValueError("Message cannot be defined as None")

        self._logger.exception(self._format(message))

    def _format(self, message: str) -> str:
        UNKNOWN = "unknown"
        if message is None:
            raise ValueError("Message cannot be defined as None")

        if not self._context:
            return message

        run_id = self._context.get("run_id", UNKNOWN)
        git = self._context.get("git_commit_hash", UNKNOWN)
        pipeline = self._context.get("pipeline_name", UNKNOWN)
        step = self._context.get("step")

        prefix = f"run_id={run_id} git={git} pipeline={pipeline}"

        if step is not None:
            prefix = f"{prefix} [{step}]"

        return f"{prefix} {message}"

    def _should_log(self, message_verbosity: VerbosityLevel) -> bool:
        return VERBOSITY_ORDER[message_verbosity] <= VERBOSITY_ORDER[self._verbosity]
