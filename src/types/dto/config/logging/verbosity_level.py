from enum import Enum


class VerbosityLevel(str, Enum):
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
