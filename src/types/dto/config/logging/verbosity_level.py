from enum import StrEnum


class VerbosityLevel(StrEnum):
    QUIET = "quiet"
    NORMAL = "normal"
    DETAILED = "detailed"
    TRACE = "trace"


VERBOSITY_ORDER = {
    VerbosityLevel.QUIET: 1,
    VerbosityLevel.NORMAL: 2,
    VerbosityLevel.DETAILED: 3,
    VerbosityLevel.TRACE: 4,
}
