@dataclass(slots=True, frozen=True)
class PipelineStepMeta:
    """
    e.g. 'validation' or 'preprocessing'.
    """
    step_name: str

    """
    Name of current implementation, e. g. 'PydanticConfigValidator'
    or 'MNEPreprocessor'.
    """
    implementation_name: str

    """
    Time, when step started.
    """
    started_at_utc: datetime

    """
    Time, when step terminated.
    """
    finished_at_utc: datetime

    """
    Optional meta-data.
    """
    extra: dict[str, Any] = field(default_factory=dict)
