from dataclasses import dataclass, field

from src.types.dto.save_artifacts.artifact_ref import ArtifactRef


@dataclass(frozen=True)
class SavedArtifactsDTO:
    artifacts: list[ArtifactRef] = field(default_factory=list)
