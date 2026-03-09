@dataclass(slots=True, frozen=True)
class ValidationIssue:
    
    code: str
    message: str
    location: str | None = None
    severity: str = "error"

