from enum import IntEnum


class VerbosityLevel(IntEnum):
    QUIET = 0
    NORMAL = 1
    DETAILED = 2
    TRACE = 3

VERBOSITY_ORDER = {
    VerbosityLevel.QUIET: 1,
    VerbosityLevel.NORMAL: 2,
    VerbosityLevel.DETAILED: 3,
    VerbosityLevel.TRACE: 4,
}