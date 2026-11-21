"""Download inputs for MOABB smoke tests; exit 75 only on a network outage."""

from pathlib import Path

import moabb.datasets as moabb_datasets
from hydra import compose, initialize_config_dir
from moabb.datasets.base import BaseDataset
from omegaconf import OmegaConf
from requests.exceptions import ConnectionError, HTTPError, Timeout

from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig


def prepare_moabb_data(configuration_directory: Path) -> bool:
    """Return False on a download outage; propagate configuration and data errors."""
    for source in load_smoke_sources(configuration_directory):
        if not download_source(source):
            return False
    return True


def load_smoke_sources(configuration_directory: Path) -> list[ExternalDatasetConfig]:
    configuration_paths = sorted((configuration_directory / "test/smoke/moabb").glob("*.yaml"))
    if not configuration_paths:
        raise ValueError("No MOABB smoke configurations found")

    sources: list[ExternalDatasetConfig] = []
    with initialize_config_dir(config_dir=str(configuration_directory), version_base=None):
        for configuration_path in configuration_paths:
            configuration_name = configuration_path.relative_to(configuration_directory).with_suffix("").as_posix()
            configuration = compose(config_name=configuration_name)
            source = ExternalDatasetConfig.model_validate(OmegaConf.to_container(configuration.source, resolve=True))
            sources.append(source)
    return sources


def download_source(source: ExternalDatasetConfig) -> bool:
    """Prepare raw files only; return False for connection or temporary HTTP errors."""
    dataset_class = getattr(moabb_datasets, source.name)
    dataset = dataset_class()
    if not isinstance(dataset, BaseDataset):
        raise TypeError(f"{source.name} is not a MOABB dataset")

    try:
        for subject_id in source.subject_ids:
            dataset.data_path(subject=subject_id)
    except (ConnectionError, Timeout) as error:
        print(f"MOABB download unavailable: {error}", flush=True)
        return False
    except HTTPError as error:
        if not is_temporary_http_failure(error):
            raise
        print(f"MOABB download unavailable: {error}", flush=True)
        return False
    return True


def is_temporary_http_failure(error: HTTPError) -> bool:
    if error.response is None:
        return False
    status_code = error.response.status_code
    return status_code in (408, 429) or 500 <= status_code <= 599


if __name__ == "__main__":
    is_available = prepare_moabb_data(Path(__file__).resolve().parents[1] / "configs")
    raise SystemExit(0 if is_available else 75)
