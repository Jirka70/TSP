"""Declaration of the experiment's methodological protocol.

The protocol states *what question the experiment answers* (which evaluation
scheme is used, i.e. what is new in the test data, whether results are claimed for offline or online
use) and *how strictly* the methodology rules are enforced (``rigor``). It is consumed by the methodology
validator (``src/validation/methodology``), not by the pipeline stages.
"""

from enum import Enum

from pydantic import BaseModel, Field


class EvaluationScheme(str, Enum):
    """What is new (never seen in training) in the test data.

    Names follow MOABB evaluations (WithinSessionEvaluation, CrossSessionEvaluation,
    CrossSubjectEvaluation) and the split configs ``split=evaulation/moabb_*``.
    """

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
        evaluation_scheme: What is new in the test data (new trials / session / subject / dataset).
        usage: Whether results are claimed for offline, simulated online or online use.
        rigor: Strictness level of the methodology validation.
    """

    task: str = Field(min_length=1)
    evaluation_scheme: EvaluationScheme
    usage: Usage = Usage.OFFLINE
    rigor: RigorLevel = RigorLevel.STANDARD
