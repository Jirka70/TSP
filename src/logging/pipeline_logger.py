import logging

from src.types.dto.config.logging.verbosity_level import VerbosityLevel


type PipelineLoggerContext = dict[str, str] | None

class PipelineLogger:
    def __init__(self,
                 logger: logging.Logger,
                 verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
                 ctx: PipelineLoggerContext = None) -> None:
        if ctx is None:
            ctx = {}
        self._logger = logger
        self._verbosity = verbosity
        self._context = ctx or {}

    def info(self, message: str, verbosity: VerbosityLevel = VerbosityLevel.NORMAL) -> None:
        if message is None:
            raise ValueError("Message cannot be defined as None")

        if verbosity <= self._verbosity:
            self._logger.info(self._format(message))

    def debug(self, message: str, verbosity: VerbosityLevel = VerbosityLevel.NORMAL) -> None:
        if message is None:
            raise ValueError("Message cannot be defined as None")

        if verbosity <= self._verbosity:
            self._logger.info(self._format(message))

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

    def _format(self, message) -> str:
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

