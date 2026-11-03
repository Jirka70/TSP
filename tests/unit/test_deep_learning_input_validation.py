from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import mne
import numpy
import pytest

from src.impl.model.deep_learning.eegnet.trainer.final_eegnet_trainer import FinalEEGNetTrainer
from src.impl.model.deep_learning.validation.validate_dl_input import validate_dl_input
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from tests.unit.deep_learning_fixtures import create_dataset


@pytest.fixture
def input_data():
    return FinalTrainingInputDTO(
        config=EEGNetConfig.model_construct(n_classes=2),
        training_data=create_dataset(1),
        validation_data=create_dataset(2),
    )


def _get_dataset_field_name(dataset_name):
    return {"train": "training_data", "val": "validation_data"}[dataset_name]


def _replace_recording(input_data, dataset_name, **changes):
    recording = replace(getattr(input_data, _get_dataset_field_name(dataset_name)).data[0], **changes)
    return replace(input_data, **{_get_dataset_field_name(dataset_name): EpochPreprocessedDTO(data=[recording])})


def _copy_metadata(input_data, dataset_name):
    return deepcopy(getattr(input_data, _get_dataset_field_name(dataset_name)).data[0].metadata)


def _replace_metadata(input_data, dataset_name, metadata):
    return _replace_recording(input_data, dataset_name, metadata=metadata)


def test_validates_all_datasets_without_changing_input_arrays(input_data):
    validated = validate_dl_input(input_data)
    assert validated.class_counts == {name: {0: 1, 1: 1} for name in ("train", "val")}
    assert validated.warnings == []
    assert not numpy.shares_memory(validated.training.x, input_data.training_data.data[0].data)
    assert validated.validation.sample_count == 2


@pytest.mark.parametrize("shape", [(2, 32), (2, 0, 32), (2, 2, 0), (2, 3, 32), (2, 2, 31)])
def test_rejects_invalid_signal_dimensions(input_data, shape):
    invalid = _replace_recording(input_data, "val", data=numpy.ones(shape))
    with pytest.raises(ValueError, match="val, recording 0"):
        validate_dl_input(invalid)


@pytest.mark.parametrize("value", [numpy.nan, numpy.inf, -numpy.inf, 1e100])
def test_rejects_invalid_signal_values_before_or_after_precision_conversion(input_data, value):
    signal = numpy.full((2, 2, 32), value)
    with pytest.raises(ValueError, match="val, recording 0:.*(NaN|infinity)"):
        validate_dl_input(_replace_recording(input_data, "val", data=signal))


@pytest.mark.parametrize("data_type", [numpy.complex64, object, bool])
def test_rejects_non_real_signal_arrays(input_data, data_type):
    signal = numpy.ones((2, 2, 32), dtype=data_type)
    with pytest.raises(ValueError, match="real numbers"):
        validate_dl_input(_replace_recording(input_data, "train", data=signal))


@pytest.mark.parametrize("labels", [[0], [[0, 1]], [0.0, 1.0]])
def test_rejects_invalid_labels(input_data, labels):
    metadata = _copy_metadata(input_data, "train")
    metadata["labels"] = numpy.asarray(labels)
    with pytest.raises(ValueError, match="train, recording 0"):
        validate_dl_input(_replace_metadata(input_data, "train", metadata))


def test_checks_each_recording_before_concatenation(input_data):
    first = input_data.training_data.data[0]
    second = deepcopy(first)
    first.metadata["labels"] = numpy.array([0])
    second.metadata["labels"] = numpy.array([0, 1, 1])
    invalid = replace(input_data, training_data=EpochPreprocessedDTO(data=[first, second]))
    with pytest.raises(ValueError, match="train, recording 0:.*counts differ"):
        validate_dl_input(invalid)


def test_rejects_missing_training_classes(input_data):
    metadata = _copy_metadata(input_data, "train")
    metadata["labels"] = numpy.array([0, 0])
    with pytest.raises(ValueError, match="Training class count differs"):
        validate_dl_input(_replace_metadata(input_data, "train", metadata))


def test_reports_missing_evaluation_classes_without_rejecting_known_labels(input_data):
    metadata = _copy_metadata(input_data, "val")
    metadata["labels"] = numpy.array([0, 0])
    validated = validate_dl_input(_replace_metadata(input_data, "val", metadata))
    assert validated.warnings == ["val: no examples for classes [1]."]


def test_rejects_configured_class_count_mismatch(input_data):
    invalid = replace(input_data, config=EEGNetConfig.model_construct(n_classes=3))
    with pytest.raises(ValueError, match="Training class count differs"):
        validate_dl_input(invalid)


def test_validates_native_epoch_container_metadata(input_data):
    information = mne.create_info(["C3", "C4"], 32.0, ch_types="eeg")
    events = numpy.array([[0, 0, 0], [32, 0, 1]])
    epochs = mne.EpochsArray(input_data.training_data.data[0].data, information, events=events, verbose=False)
    validate_dl_input(_replace_recording(input_data, "train", data=epochs))


def test_rejects_invalid_input_before_creating_or_training_model(input_data):
    metadata = _copy_metadata(input_data, "val")
    del metadata["labels"]
    invalid = _replace_metadata(input_data, "val", metadata)
    factory = Mock()
    with pytest.raises(ValueError, match="val, recording 0"):
        FinalEEGNetTrainer(factory).run(invalid, SimpleNamespace(logger=Mock()))
    factory.create.assert_not_called()


def test_rejects_validation_class_absent_from_training(input_data):
    metadata = _copy_metadata(input_data, "val")
    metadata["labels"] = numpy.array([0, 2])
    with pytest.raises(ValueError, match="Validation contains classes absent from train"):
        validate_dl_input(_replace_metadata(input_data, "val", metadata))


def test_accepts_consistent_nonconsecutive_class_identifiers(input_data):
    for dataset in (input_data.training_data, input_data.validation_data):
        dataset.data[0].metadata["labels"] = numpy.array([3, 7])
    validated = validate_dl_input(input_data)
    assert validated.class_counts == {name: {3: 1, 7: 1} for name in ("train", "val")}


def test_does_not_require_or_interpret_signal_description_or_origin_metadata(input_data):
    input_data.training_data.data[0].metadata.update(signal_specification=None, epoch_origins=[])
    validated = validate_dl_input(input_data)
    assert validated.training.sample_count == 2
    assert "signal_specification" not in type(validated).model_fields


def test_native_epochs_do_not_require_label_metadata(input_data):
    information = mne.create_info(["C3", "C4"], 32.0, ch_types="eeg")
    events = numpy.array([[0, 0, 0], [32, 0, 1]])
    epochs = mne.EpochsArray(input_data.training_data.data[0].data, information, events=events, verbose=False)
    metadata = {}
    replacement = _replace_recording(input_data, "train", data=epochs, metadata=metadata)
    assert validate_dl_input(replacement).training.sample_count == 2
