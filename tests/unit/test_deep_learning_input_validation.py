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
from src.types.dto.model.dl_input_dto import DLInputDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.split.dataset_split_dto import FoldDTO
from tests.unit.deep_learning_fixtures import create_dataset


@pytest.fixture
def input_data():
    return DLInputDTO(
        config=EEGNetConfig.model_construct(n_classes=2),
        train=create_dataset(1),
        val=create_dataset(2),
        test=create_dataset(3),
    )


def _replace_recording(input_data, dataset_name, **changes):
    recording = replace(getattr(input_data, dataset_name).data[0], **changes)
    return replace(input_data, **{dataset_name: EpochPreprocessedDTO(data=[recording])})


def _copy_metadata(input_data, dataset_name):
    return deepcopy(getattr(input_data, dataset_name).data[0].metadata)


def _replace_metadata(input_data, dataset_name, metadata):
    return _replace_recording(input_data, dataset_name, metadata=metadata)


def test_validates_all_datasets_without_changing_input_arrays(input_data):
    validated = validate_dl_input(input_data)
    assert validated.class_counts == {name: {0: 1, 1: 1} for name in ("train", "val", "test")}
    assert validated.warnings == []
    assert not numpy.shares_memory(validated.training.x, input_data.train.data[0].data)
    assert validated.test.sample_count == 2


@pytest.mark.parametrize("dataset_name", ["train", "val", "test"])
def test_rejects_missing_provenance_with_dataset_and_recording_location(input_data, dataset_name):
    metadata = _copy_metadata(input_data, dataset_name)
    del metadata["epoch_origins"]
    with pytest.raises(ValueError, match=f"{dataset_name}, recording 0:.*metadata"):
        validate_dl_input(_replace_metadata(input_data, dataset_name, metadata))


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


@pytest.mark.parametrize("labels", [[0], [[0, 1]], [0.0, 1.0], [0, 2]])
def test_rejects_invalid_labels(input_data, labels):
    metadata = _copy_metadata(input_data, "train")
    metadata["labels"] = numpy.asarray(labels)
    with pytest.raises(ValueError, match="train, recording 0"):
        validate_dl_input(_replace_metadata(input_data, "train", metadata))


def test_checks_each_recording_before_concatenation(input_data):
    first = input_data.train.data[0]
    second = deepcopy(first)
    first.metadata["labels"] = numpy.array([0])
    second.metadata["labels"] = numpy.array([0, 1, 1])
    invalid = replace(input_data, train=EpochPreprocessedDTO(data=[first, second]))
    with pytest.raises(ValueError, match="train, recording 0:.*counts differ"):
        validate_dl_input(invalid)


def test_rejects_missing_training_classes(input_data):
    metadata = _copy_metadata(input_data, "train")
    metadata["labels"] = numpy.array([0, 0])
    with pytest.raises(ValueError, match="Train lacks declared classes"):
        validate_dl_input(_replace_metadata(input_data, "train", metadata))


def test_reports_missing_evaluation_classes_without_rejecting_known_labels(input_data):
    metadata = _copy_metadata(input_data, "val")
    metadata["labels"] = numpy.array([0, 0])
    validated = validate_dl_input(_replace_metadata(input_data, "val", metadata))
    assert validated.warnings == ["val: no examples for classes [1]."]


def test_rejects_configured_class_count_mismatch(input_data):
    invalid = replace(input_data, config=EEGNetConfig.model_construct(n_classes=3))
    with pytest.raises(ValueError, match="Declared class count"):
        validate_dl_input(invalid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("channel_names", ["C4", "C3"]),
        ("sampling_frequency", 64.0),
        ("unit", "uV"),
        ("label_encoding_identifier", "another-encoding"),
        ("preprocessing_identifier", "another-filter"),
        ("window_start_seconds", -1.0),
    ],
)
def test_rejects_inconsistent_signal_specifications(input_data, field, value):
    metadata = _copy_metadata(input_data, "test")
    metadata["signal_specification"][field] = value
    with pytest.raises(ValueError, match="test, recording 0: signal specification differs"):
        validate_dl_input(_replace_metadata(input_data, "test", metadata))


@pytest.mark.parametrize("dataset_name", ["val", "test"])
def test_rejects_augmented_evaluation_epochs(input_data, dataset_name):
    metadata = _copy_metadata(input_data, dataset_name)
    metadata["epoch_origins"][0]["augmented"] = True
    with pytest.raises(ValueError, match="contains training augmentation"):
        validate_dl_input(_replace_metadata(input_data, dataset_name, metadata))


def test_allows_training_copy_with_original_provenance(input_data):
    metadata = _copy_metadata(input_data, "train")
    metadata["epoch_origins"][1] = deepcopy(metadata["epoch_origins"][0])
    metadata["epoch_origins"][1]["augmented"] = True
    validate_dl_input(_replace_metadata(input_data, "train", metadata))


def test_rejects_original_trial_shared_between_datasets(input_data):
    metadata = _copy_metadata(input_data, "test")
    metadata["epoch_origins"][0]["roots"] = deepcopy(input_data.train.data[0].metadata["epoch_origins"][0]["roots"])
    with pytest.raises(ValueError, match="shared trial_identifier"):
        validate_dl_input(_replace_metadata(input_data, "test", metadata))


@pytest.mark.parametrize("start,stop", [(1000, 1032), (1010, 1015), (999, 1100)])
def test_rejects_cross_dataset_signal_overlap_even_with_different_trial_identifiers(input_data, start, stop):
    metadata = _copy_metadata(input_data, "val")
    metadata["epoch_origins"][0]["roots"][0].update(start_sample=start, stop_sample=stop)
    with pytest.raises(ValueError, match="intervals overlap"):
        validate_dl_input(_replace_metadata(input_data, "val", metadata))


def test_accepts_adjacent_nonoverlapping_intervals(input_data):
    metadata = _copy_metadata(input_data, "val")
    metadata["epoch_origins"][0]["roots"][0].update(start_sample=1064, stop_sample=1096)
    validate_dl_input(_replace_metadata(input_data, "val", metadata))


@pytest.mark.parametrize("protocol,field", [("session", "session_identifier"), ("subject", "subject_identifier")])
def test_enforces_declared_group_separation(input_data, protocol, field):
    for dataset_name in ("train", "val", "test"):
        getattr(input_data, dataset_name).data[0].metadata["signal_specification"]["split_protocol"] = protocol
    with pytest.raises(ValueError, match=f"shared {field}"):
        validate_dl_input(input_data)


def test_rejects_preprocessing_fitted_on_evaluation_trial(input_data):
    test_trial = input_data.test.data[0].metadata["epoch_origins"][0]["roots"][0]["trial_identifier"]
    for dataset_name in ("train", "val", "test"):
        description = getattr(input_data, dataset_name).data[0].metadata["signal_specification"]
        description["preprocessing_fit_trial_identifiers"] = [test_trial]
    with pytest.raises(ValueError, match="Preprocessing was fitted on trials outside train"):
        validate_dl_input(input_data)


def test_rejects_conflicting_source_owners(input_data):
    metadata = _copy_metadata(input_data, "test")
    metadata["epoch_origins"][0]["roots"][0]["session_identifier"] = "different-session"
    with pytest.raises(ValueError, match="Inconsistent owner of source"):
        validate_dl_input(_replace_metadata(input_data, "test", metadata))


def test_validates_native_epoch_container_metadata(input_data):
    information = mne.create_info(["C3", "C4"], 32.0, ch_types="eeg")
    events = numpy.array([[0, 0, 0], [32, 0, 1]])
    epochs = mne.EpochsArray(input_data.train.data[0].data, information, events=events, verbose=False)
    validate_dl_input(_replace_recording(input_data, "train", data=epochs))
    epochs = mne.EpochsArray(
        input_data.train.data[0].data,
        information,
        events=events,
        tmin=-1.0,
        verbose=False,
    )
    with pytest.raises(ValueError, match="window start differs"):
        validate_dl_input(_replace_recording(input_data, "train", data=epochs))


def test_rejects_invalid_input_before_creating_or_training_model(input_data):
    metadata = _copy_metadata(input_data, "test")
    del metadata["epoch_origins"]
    invalid = _replace_metadata(input_data, "test", metadata)
    final_input = FinalTrainingInputDTO(
        config=input_data.config,
        training_data=input_data.train,
        folds=[FoldDTO(fold_idx=0, train_data=invalid.train, test_data=invalid.test)],
        validation_data=invalid.val,
    )
    factory = Mock()
    with pytest.raises(ValueError, match="test, recording 0"):
        FinalEEGNetTrainer(factory).run(final_input, SimpleNamespace(logger=Mock()))
    factory.create.assert_not_called()
