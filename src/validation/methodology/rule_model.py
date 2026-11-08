"""Schema of the methodology rules catalog (``configs/methodology/rules.yaml``).

Each rule describes one methodological error that must not happen, how it is
detected (a declarative condition over the config, or a registered Python
check), how certain the detection is, and which action is taken at each
rigor level.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.types.dto.config.experiment_config import Mode
from src.types.dto.config.protocol_config import RigorLevel


class RuleAction(str, Enum):
    """What happens when a rule is violated."""

    OFF = "off"
    """Rule is not evaluated at all."""

    INFO = "info"
    """Violation is reported as information."""

    WARN = "warn"
    """Violation is reported as a warning; the experiment continues."""

    ERROR = "error"
    """Violation blocks the experiment."""


class RulePhase(str, Enum):
    """When a rule can be evaluated."""

    STATIC = "static"
    """Before the run, from the experiment configuration only."""

    RUNTIME = "runtime"
    """During the run, from the actual data (e.g. after splitting). Not wired into pipelines yet."""


class Certainty(str, Enum):
    """How reliable the detection is."""

    PROVEN = "proven"
    """The violation is demonstrably present when the rule fires."""

    SUSPECTED = "suspected"
    """The violation is likely, but depends on facts the validator cannot see."""


DEFAULT_SEVERITY: dict[Certainty, dict[RigorLevel, RuleAction]] = {
    Certainty.PROVEN: {
        RigorLevel.EXPLORATORY: RuleAction.WARN,
        RigorLevel.STANDARD: RuleAction.ERROR,
        RigorLevel.PUBLICATION: RuleAction.ERROR,
    },
    Certainty.SUSPECTED: {
        RigorLevel.EXPLORATORY: RuleAction.INFO,
        RigorLevel.STANDARD: RuleAction.WARN,
        RigorLevel.PUBLICATION: RuleAction.ERROR,
    },
}
"""Action per rigor level used when a rule does not override it."""


_OPERATORS: tuple[str, ...] = ("eq", "ne", "in_", "not_in", "exists", "gt", "ge", "lt", "le", "contains")
_COMPOSITES: tuple[str, ...] = ("all_", "any_", "not_")


class Condition(BaseModel):
    """Declarative condition over the experiment configuration.

    Either a composite node (exactly one of ``all`` / ``any`` / ``not``) or a
    leaf (``path`` + exactly one operator). ``path`` is a dot-separated path
    into the dumped config; integer segments index into lists
    (e.g. ``paradigm.window.baseline.1``).

    Leaf semantics when ``path`` does not exist: ``exists: false`` is satisfied,
    every other operator is not satisfied.

    Examples (YAML)::

        {path: epoch_preprocessing.csp.enabled, eq: true}
        {path: protocol.generalization, in: [within_session, within_subject]}
        {path: evaluation.metrics, contains: balanced_accuracy}
        all: [{...}, {not: {...}}]
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    all_: list[Condition] | None = Field(default=None, alias="all")
    any_: list[Condition] | None = Field(default=None, alias="any")
    not_: Condition | None = Field(default=None, alias="not")

    path: str | None = None
    eq: Any = None
    ne: Any = None
    in_: list[Any] | None = Field(default=None, alias="in")
    not_in: list[Any] | None = None
    exists: bool | None = None
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    contains: Any = None

    @model_validator(mode="after")
    def validate_shape(self) -> Condition:
        """Ensures the node is either one composite or a path with exactly one operator."""
        composites = [name for name in _COMPOSITES if name in self.model_fields_set]
        operators = [name for name in _OPERATORS if name in self.model_fields_set]

        if composites:
            if len(composites) > 1 or self.path is not None or operators:
                raise ValueError("A condition must be exactly one of 'all', 'any', 'not', or a 'path' with one operator.")
            return self

        if self.path is None:
            raise ValueError("A leaf condition requires 'path'.")
        if len(operators) != 1:
            raise ValueError(f"Condition on '{self.path}' must have exactly one operator from {[o.rstrip('_') for o in _OPERATORS]} (got {operators}).")
        return self

    @property
    def operator(self) -> str | None:
        """Name of the leaf operator (``None`` for composite nodes)."""
        for name in _OPERATORS:
            if name in self.model_fields_set:
                return name
        return None


class MethodologyRule(BaseModel):
    """One rule of the methodology catalog.

    Attributes:
        id: Unique rule identifier (e.g. ``LEAK-001``).
        title: Short human-readable name of the violation.
        description: What exactly is wrong when the rule fires.
        section: Reference to the methodology document section.
        scenario: Optional reference to a demonstration scenario (A-F).
        phase: Whether the rule is checked statically (config) or at runtime (data).
        certainty: Whether the violation is proven or only suspected.
        modes: Pipeline modes the rule applies to (``None`` = all modes).
        when: Declarative condition; the rule fires when it is satisfied.
        check: Name of a registered Python check (alternative to ``when``).
        params: Parameters passed to the Python check.
        severity: Per-rigor action overrides (missing levels use ``DEFAULT_SEVERITY``).
        impact: Possible impact on the reported results.
        remediation: How to fix the experiment.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    section: str | None = None
    scenario: str | None = None
    phase: RulePhase = RulePhase.STATIC
    certainty: Certainty
    modes: list[Mode] | None = None

    when: Condition | None = None
    check: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    severity: dict[RigorLevel, RuleAction] = Field(default_factory=dict)

    impact: str = ""
    remediation: str = ""

    @model_validator(mode="after")
    def validate_detection(self) -> MethodologyRule:
        """Ensures exactly one detection mechanism and fills default severities."""
        if (self.when is None) == (self.check is None):
            raise ValueError(f"Rule '{self.id}' must define exactly one of 'when' or 'check'.")
        if self.when is not None and self.phase != RulePhase.STATIC:
            raise ValueError(f"Rule '{self.id}': declarative 'when' conditions are only supported for static rules.")
        if self.params and self.check is None:
            raise ValueError(f"Rule '{self.id}': 'params' are only used by Python checks.")

        defaults = DEFAULT_SEVERITY[self.certainty]
        self.severity = {level: self.severity.get(level, defaults[level]) for level in RigorLevel}
        return self

    def action_for(self, rigor: RigorLevel) -> RuleAction:
        """Returns the action taken by this rule at the given rigor level."""
        return self.severity[rigor]

    def applies_to(self, mode: Mode) -> bool:
        """Whether the rule applies to the given pipeline mode."""
        return self.modes is None or Mode(mode) in self.modes


class RulesCatalog(BaseModel):
    """Root of the rules YAML file."""

    model_config = ConfigDict(extra="forbid")

    version: str
    rules: list[MethodologyRule]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> RulesCatalog:
        """Ensures rule ids are unique."""
        seen: set[str] = set()
        duplicates = {rule.id for rule in self.rules if rule.id in seen or seen.add(rule.id)}
        if duplicates:
            raise ValueError(f"Duplicate methodology rule ids: {sorted(duplicates)}")
        return self
