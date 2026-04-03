from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class PipelineStepMeta:
    step_name: str
    """
    e.g. 'validation' or 'epoch_preprocessing'.
    """

    implementation_name: str
    """
    Name of current implementation, e. g. 'PydanticConfigValidator'
    or 'MNEPreprocessor'.
    """

    started_at_utc: datetime
    """
    Time, when step started.
    """

    finished_at_utc: datetime
    """
    Time, when step terminated.
    """

    extra: dict[str, Any] = field(default_factory=dict)
    """
    Optional meta-data.
    """
