T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class StepResult(Generic[T]):
    """
    Jednotný výstup pipeline kroku.

    T = hlavní typ dat, která daný krok produkuje.

    Příklady:
    - StepResult[ValidatedConfig]
    - StepResult[PreprocessedData]
    - StepResult[EpochingData]
    """

    data: T
    """
    Hlavní výstup kroku.
    """

    artifacts: list[ArtifactRef] = field(default_factory=list)
    """
    Vedlejší artefakty vzniklé během kroku.
    Např. uložený soubor, graf, log, export.
    """

    meta: PipelineStepMeta | None = None
    """
    Metadata o provedení kroku.
    Např. název kroku, implementace, doba běhu.
    """

    warnings: list[str] = field(default_factory=list)
    """
    Ne-fatální upozornění vzniklá během kroku.
    """