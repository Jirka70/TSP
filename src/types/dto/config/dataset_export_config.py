from typing import Literal

from pydantic import BaseModel


class DatasetExportConfig(BaseModel):
    """
    Configuration for exporting the augmented dataset.
    """

    backend: Literal["fif", "none"] = "none"
    enabled: bool = False
