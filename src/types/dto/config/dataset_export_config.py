from typing import Literal

from pydantic import BaseModel


class DatasetExportConfigDefault(BaseModel):
    """
    Configuration for exporting the augmented dataset.
    """

    backend: Literal["default"]


class DatasetExportConfigNone(BaseModel):
    """
    Configuration for disabling dataset export.
    """

    backend: Literal["none"]


DatasetExportConfig = DatasetExportConfigDefault | DatasetExportConfigNone
