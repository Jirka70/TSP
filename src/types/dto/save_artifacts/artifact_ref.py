from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    """Reference on saved artifact of current experiment."""

    name: str
    """
    Aritfact name
    E.g. 'trained_model', 'metrics', 'config_snapshot'.
    """

    path: Path
    """
    Path to file on disk
    """

    kind: str
    """
    Artifact type
    E.g. 'model', 'metrics', 'config', 'history', 'plot'.
    """

    metadata: dict[str, Any] | None = None
