from dataclasses import dataclass, field
from typing import TypeVar, Generic

from src.pipeline.context.pipeline_step_meta import PipelineStepMeta

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class StepResult(Generic[T]):
    data: T
    """
    Main result of the step.
    """

    meta: PipelineStepMeta | None = None
    """
    Metadata about step execution
    """

    warnings: list[str] = field(default_factory=list)
    """
    Warning messages caused during step
    """
