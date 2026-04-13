from dataclasses import dataclass

from src.types.dto.config.paradigm_config import ParadigmConfig
from src.types.dto.load.recording import RecordingDTO


@dataclass(frozen=True)
class ParadigmInputDTO:
    """
    A data transfer object carrying the input signal for a paradigm-based epoch_preprocessing routine.

    This DTO is used to package the signal data before it is transformed
    according to the specific requirements of the current paradigm. Being
    immutable, it prevents any unintended modifications during the pipeline
    execution.

    Attributes:
        paradigm_config (ParadigmConfig): The paradigm configuration.
        signal (RawPreprocessedDTO): The preprocessed raw signal to segment into epochs.
    """

    paradigm_preprocessing_config: ParadigmConfig

    data: list[RecordingDTO]
