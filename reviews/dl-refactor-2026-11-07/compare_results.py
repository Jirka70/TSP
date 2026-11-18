"""Compare inputs, training histories, network tensors and test predictions."""

import json
from pathlib import Path

import numpy
import torch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/dl-refactor-2026-11-07"
PREVIOUS_OUTPUT = ROOT / "outputs/dl-performance-2026-11-04"


def read_json(path: Path) -> dict[str, object]:
    contents: object = json.loads(path.read_text())
    if not isinstance(contents, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return dict(contents)


def read_network_state(directory: Path) -> dict[str, torch.Tensor]:
    checkpoint_paths = list((directory / "artifacts").glob("*/models/eegnet.pt"))
    if len(checkpoint_paths) != 1:
        raise ValueError(f"Expected one checkpoint in {directory}.")
    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=True)
    network_state: object = checkpoint["model_state"]["network_state_dict"]
    if not isinstance(network_state, dict):
        raise TypeError("Expected a network state dictionary.")
    parameters: dict[str, torch.Tensor] = {}
    for parameter_name, parameter in network_state.items():
        is_valid_parameter = isinstance(parameter_name, str) and isinstance(parameter, torch.Tensor)
        if not is_valid_parameter:
            raise TypeError("Expected named tensor parameters.")
        parameters[parameter_name] = parameter
    return parameters


def are_network_states_equal(first_directory: Path, second_directory: Path) -> bool:
    first_state = read_network_state(first_directory)
    second_state = read_network_state(second_directory)
    if first_state.keys() != second_state.keys():
        return False
    return all(torch.equal(parameter, second_state[name]) for name, parameter in first_state.items())


def compare_predictions(first_directory: Path, second_directory: Path) -> dict[str, object]:
    with (
        numpy.load(first_directory / "test_predictions.npz") as first_predictions,
        numpy.load(second_directory / "test_predictions.npz") as second_predictions,
    ):
        are_labels_equal = numpy.array_equal(first_predictions["labels"], second_predictions["labels"])
        are_probabilities_equal = numpy.array_equal(
            first_predictions["probabilities"], second_predictions["probabilities"]
        )
        maximum_probability_difference = float(
            numpy.max(numpy.abs(first_predictions["probabilities"] - second_predictions["probabilities"]))
        )
    return {
        "are_labels_equal": bool(are_labels_equal),
        "are_probabilities_equal": bool(are_probabilities_equal),
        "maximum_probability_difference": maximum_probability_difference,
    }


def compare_saved_reference_predictions(directory: Path) -> dict[str, object]:
    with (
        numpy.load(directory / "test_predictions.npz") as predictions,
        numpy.load(directory / "reference_predictions.npz") as reference_predictions,
    ):
        are_labels_equal = numpy.array_equal(predictions["labels"], reference_predictions["labels"])
        are_probabilities_equal = numpy.array_equal(
            predictions["probabilities"], reference_predictions["probabilities"]
        )
        maximum_probability_difference = float(
            numpy.max(numpy.abs(predictions["probabilities"] - reference_predictions["probabilities"]))
        )
    return {
        "are_labels_equal": bool(are_labels_equal),
        "are_probabilities_equal": bool(are_probabilities_equal),
        "maximum_probability_difference": maximum_probability_difference,
    }


def comparable_history(history: object) -> dict[str, object]:
    if not isinstance(history, dict):
        raise TypeError("Training history must be a dictionary.")
    validation_metrics = history.get("validation_metrics")
    if not isinstance(validation_metrics, dict):
        raise TypeError("Validation metrics must be a dictionary.")
    if "fold_accuracy" in validation_metrics:
        if validation_metrics.get("validation_accuracy", []) != []:
            raise ValueError("The historical validation metric mapping is ambiguous.")
        validation_accuracy = validation_metrics["fold_accuracy"]
    else:
        validation_accuracy = validation_metrics["validation_accuracy"]
    return {
        "train_loss": history["train_loss"],
        "validation_loss": history["validation_loss"],
        "train_metrics": history["train_metrics"],
        "validation_accuracy": validation_accuracy,
    }


def compare_case(case: str) -> dict[str, object]:
    baseline_directory = OUTPUT / "baseline" / case
    refactored_directory = OUTPUT / "refactored" / case
    previous_directory = PREVIOUS_OUTPUT
    if case == "subject-8":
        previous_directory = PREVIOUS_OUTPUT / "subject-8-held-out"
    reference_kind = "repeated_baseline"
    if not (baseline_directory / "results.json").exists():
        baseline_directory = previous_directory
        reference_kind = "november_4_saved_experiment"
    baseline = read_json(baseline_directory / "results.json")
    refactored = read_json(refactored_directory / "results.json")
    previous = read_json(previous_directory / "results.json")
    compared_fields = (
        "configuration",
        "versions",
        "device",
        "threads",
        "datasets",
        "best_epoch_zero_based",
        "best_validation_accuracy",
        "test_metrics",
        "confusion_matrix",
    )
    baseline_comparison = {field: baseline[field] == refactored[field] for field in compared_fields}
    previous_comparison = {field: previous[field] == refactored[field] for field in compared_fields}
    baseline_comparison["history_values"] = comparable_history(baseline["history"]) == comparable_history(
        refactored["history"]
    )
    previous_comparison["history_values"] = comparable_history(previous["history"]) == comparable_history(
        refactored["history"]
    )
    are_baseline_weights_equal = are_network_states_equal(baseline_directory, refactored_directory)
    are_previous_weights_equal = are_network_states_equal(previous_directory, refactored_directory)
    if reference_kind == "repeated_baseline":
        prediction_comparison = compare_predictions(baseline_directory, refactored_directory)
    else:
        prediction_comparison = compare_saved_reference_predictions(refactored_directory)
    is_case_equal = (
        all(baseline_comparison.values())
        and all(previous_comparison.values())
        and are_baseline_weights_equal
        and are_previous_weights_equal
        and prediction_comparison["are_labels_equal"] is True
        and prediction_comparison["are_probabilities_equal"] is True
    )
    return {
        "is_case_equal": is_case_equal,
        "reference_kind": reference_kind,
        "are_raw_history_keys_equal_to_november_4": previous["history"] == refactored["history"],
        "baseline_comparison": baseline_comparison,
        "november_4_comparison": previous_comparison,
        "are_baseline_weights_equal": are_baseline_weights_equal,
        "are_november_4_weights_equal": are_previous_weights_equal,
        "test_prediction_comparison": prediction_comparison,
        "test_metrics": refactored["test_metrics"],
        "best_epoch_zero_based": refactored["best_epoch_zero_based"],
    }


if __name__ == "__main__":
    comparisons = {case: compare_case(case) for case in ("general", "subject-8")}
    comparison_path = OUTPUT / "comparison.json"
    comparison_path.write_text(json.dumps(comparisons, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))
    if not all(comparison["is_case_equal"] is True for comparison in comparisons.values()):
        raise SystemExit("The experiments differ; inspect comparison.json.")
