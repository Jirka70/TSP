from dataclasses import dataclass, field

from tsp_eeg_classification.types.dto.save_artifacts.artifact_ref import ArtifactRef


@dataclass(frozen=True)
class SavedArtifactsDTO:
    artifacts: list[ArtifactRef] = field(default_factory=list)
