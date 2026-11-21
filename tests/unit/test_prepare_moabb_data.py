"""Network fallback applies only to downloading smoke test inputs."""

from pathlib import Path
from unittest.mock import patch

import pytest
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, Timeout

from scripts.prepare_moabb_data import download_source, prepare_moabb_data
from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig


def create_source() -> ExternalDatasetConfig:
    return ExternalDatasetConfig(
        backend="external",
        name="BNCI2014_001",
        path=Path("./data/eegbci"),
        subject_ids=[1],
        session_ids=None,
        run_ids=None,
    )


def test_prepares_dataset_without_fallback() -> None:
    with patch("moabb.datasets.BNCI2014_001.data_path", return_value=[]) as download:
        assert download_source(create_source())
    download.assert_called_once_with(subject=1)


@pytest.mark.parametrize("error", [ConnectionError("offline"), Timeout("timeout")])
def test_network_outage_selects_fallback(error: Exception) -> None:
    with patch("moabb.datasets.BNCI2014_001.data_path", side_effect=error):
        assert not download_source(create_source())


@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
def test_temporary_http_failure_selects_fallback(status_code: int) -> None:
    response = Response()
    response.status_code = status_code
    error = HTTPError(response=response)
    with patch("moabb.datasets.BNCI2014_001.data_path", side_effect=error):
        assert not download_source(create_source())


@pytest.mark.parametrize("status_code", [403, 404])
def test_permanent_http_failure_still_fails(status_code: int) -> None:
    response = Response()
    response.status_code = status_code
    with patch("moabb.datasets.BNCI2014_001.data_path", side_effect=HTTPError(response=response)):
        with pytest.raises(HTTPError):
            download_source(create_source())


def test_data_error_still_fails() -> None:
    with patch("moabb.datasets.BNCI2014_001.data_path", side_effect=ValueError("corrupt input")):
        with pytest.raises(ValueError, match="corrupt input"):
            download_source(create_source())


def test_prepares_inputs_from_actual_smoke_configurations() -> None:
    configuration_directory = Path(__file__).resolve().parents[2] / "configs"
    with patch("moabb.datasets.BNCI2014_001.data_path", return_value=[]) as download:
        assert prepare_moabb_data(configuration_directory)
    assert download.call_count == 3


def test_stops_preparation_on_download_outage() -> None:
    configuration_directory = Path(__file__).resolve().parents[2] / "configs"
    with patch("moabb.datasets.BNCI2014_001.data_path", side_effect=Timeout("offline")) as download:
        assert not prepare_moabb_data(configuration_directory)
    download.assert_called_once_with(subject=1)
