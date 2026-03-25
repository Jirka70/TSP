from braindecode.datasets import MOABBDataset

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader
import moabb.datasets as moabb_datasets
import moabb.paradigms as moabb_paradigms



class MOABBDataLoader(IDataLoader):
    """
    See datasets:
    https://moabb.neurotechx.com/docs/generated/moabb.datasets.Yang2025.html
    """

    def _create_dataset(self, name: str):
        try:
            data_class = getattr(moabb_datasets, name)
            return data_class()
        except AttributeError:
            return ValueError(f"Dataset {name} was not found")

    def _create_paradigm(self, name: str):
        try:
            paradigm_class = getattr(moabb_paradigms, name)
            return paradigm_class()
        except AttributeError:
            return ValueError(f"Paradigm {name} was not found")

    def run(self, input: DatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        dataset_name: str = input.name
        paradigm_name: str = input.paradigm

        dataset = self._create_dataset(dataset_name)
        paradigm = self._create_paradigm(paradigm_name)


        return StepResult(2)
