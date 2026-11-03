"""Check declared original signal identities and disjoint dataset boundaries."""

from collections import defaultdict
from itertools import combinations

from src.impl.model.deep_learning.validation.input_schemas import OriginalSignalInterval, SignalSpecification


OriginalIntervalsByDataset = dict[str, list[OriginalSignalInterval]]


def validate_original_signal_identities(intervals_by_dataset: OriginalIntervalsByDataset) -> None:
    identity_owners: dict[tuple[str, str], tuple[str, ...]] = {}
    for intervals in intervals_by_dataset.values():
        for interval in intervals:
            _validate_single_interval_identity(interval, identity_owners)


def _validate_single_interval_identity(
    interval: OriginalSignalInterval,
    identity_owners: dict[tuple[str, str], tuple[str, ...]],
) -> None:
    subject = (interval.subject_identifier,)
    session = (*subject, interval.session_identifier)
    source = (*session, interval.source_identifier)
    _require_consistent_identity_owner(identity_owners, "session", interval.session_identifier, subject)
    _require_consistent_identity_owner(identity_owners, "source", interval.source_identifier, session)
    _require_consistent_identity_owner(identity_owners, "trial", interval.trial_identifier, source)


def _require_consistent_identity_owner(
    identity_owners: dict[tuple[str, str], tuple[str, ...]],
    identity_kind: str,
    identifier: str,
    owner: tuple[str, ...],
) -> None:
    previous_owner = identity_owners.setdefault((identity_kind, identifier), owner)
    if previous_owner != owner:
        raise ValueError(f"Inconsistent owner of {identity_kind} '{identifier}'.")


def validate_protocol_separation(
    intervals_by_dataset: OriginalIntervalsByDataset,
    specification: SignalSpecification,
) -> None:
    for first_name, second_name in combinations(intervals_by_dataset, 2):
        for field_name in _get_required_separation_fields(specification.split_protocol):
            _require_disjoint_identifiers(first_name, second_name, intervals_by_dataset, field_name)


def _get_required_separation_fields(protocol: str) -> tuple[str, ...]:
    additional_fields = {"trial": (), "session": ("session_identifier",), "subject": ("subject_identifier",)}
    return ("trial_identifier", *additional_fields[protocol])


def _require_disjoint_identifiers(
    first_name: str,
    second_name: str,
    intervals_by_dataset: OriginalIntervalsByDataset,
    field_name: str,
) -> None:
    first_identifiers = {getattr(interval, field_name) for interval in intervals_by_dataset[first_name]}
    second_identifiers = {getattr(interval, field_name) for interval in intervals_by_dataset[second_name]}
    shared_identifiers = first_identifiers & second_identifiers
    if shared_identifiers:
        raise ValueError(f"{first_name}/{second_name}: shared {field_name}: {sorted(shared_identifiers)[:3]}.")


def validate_original_intervals_do_not_overlap(intervals_by_dataset: OriginalIntervalsByDataset) -> None:
    intervals_by_source: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for dataset_name, intervals in intervals_by_dataset.items():
        for interval in intervals:
            intervals_by_source[interval.source_identifier].append(
                (interval.start_sample, interval.stop_sample, dataset_name)
            )
    for source_identifier, intervals in intervals_by_source.items():
        _validate_single_source_intervals(source_identifier, intervals)


def _validate_single_source_intervals(source_identifier: str, intervals: list[tuple[int, int, str]]) -> None:
    furthest_stops: dict[str, int] = {}
    for start_sample, stop_sample, dataset_name in sorted(intervals):
        if has_overlap_with_other_datasets(start_sample, dataset_name, furthest_stops):
            raise ValueError(
                f"{dataset_name}: original signal intervals overlap another dataset in source '{source_identifier}'."
            )
        furthest_stops[dataset_name] = max(furthest_stops.get(dataset_name, stop_sample), stop_sample)


def has_overlap_with_other_datasets(start_sample: int, dataset_name: str, furthest_stops: dict[str, int]) -> bool:
    return any(start_sample < stop for other_name, stop in furthest_stops.items() if other_name != dataset_name)


def validate_preprocessing_was_fitted_only_on_training_trials(
    intervals_by_dataset: OriginalIntervalsByDataset,
    specification: SignalSpecification,
) -> None:
    training_trials = {interval.trial_identifier for interval in intervals_by_dataset["train"]}
    invalid_trials = set(specification.preprocessing_fit_trial_identifiers) - training_trials
    if invalid_trials:
        raise ValueError(f"Preprocessing was fitted on trials outside train: {sorted(invalid_trials)[:3]}.")
