"""Declaration of the experiment's methodological protocol.

The protocol states *what question the experiment answers* (which kind of
generalization is evaluated, whether results are claimed for offline or online
use, how adaptation to new users is handled) and *how strictly* the
methodology rules are enforced (``rigor``). It is consumed by the methodology
validator (``src/validation/methodology``), not by the pipeline stages.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Generalization(str, Enum):
    """Kind of generalization the evaluation protocol is supposed to measure."""

    WITHIN_SESSION = "within_session"
    """New trials of an already seen session."""

    WITHIN_SUBJECT = "within_subject"
    """New trials of an already seen subject (sessions may be pooled)."""

    CROSS_SESSION = "cross_session"
    """A new, unseen session of an already seen subject."""

    CROSS_SUBJECT = "cross_subject"
    """A completely unseen subject."""

    CROSS_DATASET = "cross_dataset"
    """A different dataset / recording site."""


class Usage(str, Enum):
    """Claimed usage scenario of the results."""

    OFFLINE = "offline"
    SIMULATED_ONLINE = "simulated_online"
    ONLINE = "online"


class Adaptation(str, Enum):
    """Declared adaptation to the target user / session."""

    NONE = "none"
    LABELED_CALIBRATION = "labeled_calibration"
    UNLABELED_TARGET = "unlabeled_target"


class RigorLevel(str, Enum):
    """How strictly methodology rules are enforced."""

    EXPLORATORY = "exploratory"
    """Quick experiments; violations are mostly reported, rarely blocking."""

    STANDARD = "standard"
    """Default; provable violations block the experiment."""

    PUBLICATION = "publication"
    """Results meant to be reported; even suspected violations block."""


class ProtocolConfig(BaseModel):
    """Methodological protocol of the experiment.

    Attributes:
        task: Research question / task definition (e.g. "4-class motor imagery").
        generalization: Kind of generalization the experiment evaluates.
        usage: Whether results are claimed for offline, simulated online or online use.
        adaptation: Declared adaptation to the target user.
        rigor: Strictness level of the methodology validation.
    """

    task: str = Field(min_length=1)
    generalization: Generalization
    usage: Usage = Usage.OFFLINE
    adaptation: Adaptation = Adaptation.NONE
    rigor: RigorLevel = RigorLevel.STANDARD
