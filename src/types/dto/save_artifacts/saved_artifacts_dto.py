from dataclasses import dataclass, field



@dataclass(frozen=True)
class SavedArtifactsDTO:
    artifacts: list[ArtifactRef] = field(default_factory=list)