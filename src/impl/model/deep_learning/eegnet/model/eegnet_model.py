import copy
from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.impl.model.deep_learning.eegnet.model.epoch_metrics import EpochMetrics
from src.impl.model.deep_learning.validation.array_validation import (
    convert_to_array,
    copy_as_training_precision,
    has_only_finite_numbers,
    has_only_positive_dimensions,
    is_array_with_dimensions,
)
from src.impl.model.deep_learning.validation.numerical_validation import (
    validate_class_probabilities,
    validate_class_scores,
    validate_finite_gradients,
    validate_finite_loss,
    validate_finite_parameters,
)
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.model.train_history import TrainingHistory
from src.types.interfaces.model.model import IModel

PER_EPOCH_CHANNEL_NORMALIZATION = "per_epoch_channel"


class EEGNetModel(IModel):
    """
    IModel wrapper around a PyTorch EEGNet network.
    """

    def __init__(
        self, network: nn.Module, model_name: str, config: EEGNetConfig, input_shape: tuple[int, int, int]
    ) -> None:
        self._network = network
        self._model_name = model_name
        self._configuration = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._network.to(self._device)
        self._loss_function = nn.CrossEntropyLoss()
        self._optimizer: Optimizer = self._create_optimizer()

        self.best_epoch: int | None = None
        self.best_validation_accuracy: float | None = None
        self.history: TrainingHistory | None = None

        self._classes: NDArray[np.generic] | None = None
        self._class_to_index: dict[object, int] | None = None

        self.input_shape = input_shape

    def name(self) -> str:
        return self._model_name

    def fit(self, x: NDArray[np.generic], y: NDArray[np.generic]) -> None:
        """
        Basic fit required by IModel.

        Trains on one dataset for the configured number of epochs.
        No validation or external evaluation is performed.
        """

        self.initialize_training(y)
        for _ in range(self._configuration.training.epochs):
            self.train_one_epoch(x, y)

    def initialize_training(self, y_train: NDArray[np.generic]) -> None:
        if self._classes is None:
            self._fit_label_mapping(y_train)

        self.history = TrainingHistory(
            train_loss=[],
            validation_loss=[],
            train_metrics={"accuracy": []},
            validation_metrics={"validation_accuracy": []},
        )

    def train_one_epoch(self, x_train: NDArray[np.generic], y_train: NDArray[np.generic]) -> tuple[float, float]:
        if self.history is None:
            raise ValueError("Training is not initialized. Call initialize_training first.")

        encoded_training_labels = self._encode_labels(y_train)
        training_loader = self._create_loader(x_train, encoded_training_labels, shuffle=True)

        training_loss, training_accuracy = self._run_training_loader(training_loader)

        self.history.train_loss.append(training_loss)
        self.history.train_metrics["accuracy"].append(training_accuracy)

        return training_loss, training_accuracy

    def validate(self, x_val: NDArray[np.generic], y_val: NDArray[np.generic]) -> tuple[float, float]:
        if self.history is None:
            raise ValueError("Training is not initialized. Call initialize_training first.")

        encoded_validation_labels = self._encode_labels(y_val)
        validation_loader = self._create_loader(x=x_val, y=encoded_validation_labels, shuffle=False)

        validation_loss, validation_accuracy = self._run_evaluation_loader(validation_loader)

        self.history.validation_loss.append(validation_loss)
        self.history.validation_metrics["validation_accuracy"].append(validation_accuracy)

        return validation_loss, validation_accuracy

    def evaluate(self, x: NDArray[np.generic], y: NDArray[np.generic]) -> tuple[float, float]:
        if self.history is None:
            raise ValueError("Training is not initialized. Call initialize_training first.")

        encoded_labels = self._encode_labels(y)
        data_loader = self._create_loader(x, encoded_labels, shuffle=False)

        loss, accuracy = self._run_evaluation_loader(data_loader)

        return loss, accuracy

    def predict(self, x: NDArray[np.generic]) -> NDArray[np.generic]:
        probabilities = self.predict_class_probability(x)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_labels = self._decode_labels(predicted_indices)
        if predicted_labels.shape != (len(x),):
            raise ValueError("Predicted labels must have shape [number of input epochs].")
        return predicted_labels

    def predict_class_probability(self, x: NDArray[np.generic]) -> NDArray[np.generic]:
        self._network.eval()

        x_tensor = self._to_x_tensor(x)
        data_loader = DataLoader(
            TensorDataset(x_tensor),
            batch_size=self._configuration.training.batch_size,
            shuffle=False,
        )

        probability_batches: list[NDArray[np.generic]] = []

        with torch.no_grad():
            for batch_number, (batch_signal,) in enumerate(data_loader, start=1):
                batch_signal = batch_signal.to(self._device)

                context = f"Prediction batch {batch_number}"
                class_scores = self._compute_class_scores(batch_signal, context)
                probabilities = torch.softmax(class_scores, dim=1)
                validate_class_probabilities(probabilities, len(batch_signal), self._configuration.n_classes, context)

                probability_batches.append(probabilities.cpu().numpy())

        return np.concatenate(probability_batches, axis=0)

    def get_state_dict(self) -> dict[str, object]:
        return {
            "classes": self._classes.tolist() if self._classes is not None else None,
            "model_name": self._model_name,
            "input_shape": self.input_shape,
            "network_state_dict": self._network.state_dict(),
            "config": self._configuration.model_dump(),
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_validation_accuracy,
        }

    def restore_runtime_state(
        self,
        classes: list[object] | NDArray[np.generic] | None,
        best_epoch: int | None,
        best_validation_accuracy: float | None,
    ) -> None:
        self.best_epoch = best_epoch
        self.best_validation_accuracy = best_validation_accuracy

        if classes is None:
            self._classes = None
            self._class_to_index = None
            return

        restored_classes = np.asarray(classes)

        if restored_classes.size == 0:
            raise ValueError("Classes must not be empty")

        self._classes = restored_classes
        self._class_to_index = {class_label: class_index for class_index, class_label in enumerate(self._classes)}

    def get_network_state_dict(self) -> dict[str, torch.Tensor]:
        return copy.deepcopy(self._network.state_dict())

    def load_network_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._network.load_state_dict(state_dict)

    def _run_evaluation_loader(
        self,
        evaluation_loader: DataLoader[tuple[torch.Tensor, ...]],
    ) -> tuple[float, float]:
        self._network.eval()
        metrics = EpochMetrics()
        completed_epochs = len(self.history.train_loss) if self.history is not None else 0

        with torch.no_grad():
            for batch_number, (batch_signal, batch_labels) in enumerate(evaluation_loader, start=1):
                batch_signal = batch_signal.to(self._device)
                batch_labels = batch_labels.to(self._device)
                context = f"Evaluation after epoch {completed_epochs}, batch {batch_number}"
                class_scores, loss = self._compute_batch_loss(batch_signal, batch_labels, context)
                metrics.record_batch(class_scores, batch_labels, loss, self._configuration.training.batch_size)

        return metrics.averages()

    def _run_training_loader(
        self,
        training_loader: DataLoader[tuple[torch.Tensor, ...]],
    ) -> tuple[float, float]:
        self._network.train()
        metrics = EpochMetrics()
        epoch_number = len(self.history.train_loss) + 1 if self.history is not None else 1

        for batch_number, (batch_signal, batch_labels) in enumerate(training_loader, start=1):
            batch_signal = batch_signal.to(self._device)
            batch_labels = batch_labels.to(self._device)
            context = f"Training epoch {epoch_number}, batch {batch_number}"
            class_scores, loss = self._compute_batch_loss(batch_signal, batch_labels, context)
            self._update_parameters(loss, context)
            metrics.record_batch(class_scores, batch_labels, loss, self._configuration.training.batch_size)

        return metrics.averages()

    def _compute_batch_loss(
        self,
        batch_signal: torch.Tensor,
        batch_labels: torch.Tensor,
        context: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_scores = self._compute_class_scores(batch_signal, context)
        loss = self._loss_function(class_scores, batch_labels)
        validate_finite_loss(loss, context)
        return class_scores, loss

    def _compute_class_scores(self, batch_signal: torch.Tensor, context: str) -> torch.Tensor:
        class_scores: torch.Tensor = self._network(batch_signal)
        validate_class_scores(class_scores, len(batch_signal), self._configuration.n_classes, context)
        return class_scores

    def _update_parameters(self, loss: torch.Tensor, context: str) -> None:
        self._optimizer.zero_grad()
        torch.autograd.backward(loss)
        validate_finite_gradients(self._network, context)
        self._optimizer.step()
        validate_finite_parameters(self._network, context)

    def _create_loader(
        self,
        x: NDArray[np.generic],
        y: NDArray[np.generic],
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        x_tensor = self._to_x_tensor(x)
        y_tensor = torch.tensor(y, dtype=torch.long)

        is_shuffle_enabled = shuffle
        dataset = TensorDataset(x_tensor, y_tensor)

        return DataLoader(
            dataset,
            batch_size=self._configuration.training.batch_size,
            shuffle=is_shuffle_enabled,
        )

    def _normalize_input(self, signal: NDArray[np.generic]) -> NDArray[np.generic]:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            channel_means = signal.mean(axis=2, keepdims=True)
            channel_standard_deviations = signal.std(axis=2, keepdims=True)
            normalized_signal: NDArray[np.generic] = (signal - channel_means) / (channel_standard_deviations + 1e-6)
        if not has_only_finite_numbers(channel_means):
            raise ValueError("EEGNet input contains NaN or infinity after per-epoch channel normalization (mean).")
        if not has_only_finite_numbers(channel_standard_deviations):
            raise ValueError("EEGNet input contains NaN or infinity after per-epoch channel normalization (deviation).")
        if not has_only_finite_numbers(normalized_signal):
            raise ValueError("EEGNet input contains NaN or infinity after per-epoch channel normalization.")
        return normalized_signal

    def _to_x_tensor(self, x: NDArray[np.generic]) -> torch.Tensor:
        signal = copy_as_training_precision(convert_to_array(x))
        if not is_array_with_dimensions(signal, 3):
            raise ValueError(f"EEGNet expects shape [epochs, channels, samples], got {signal.shape}.")
        if not has_only_positive_dimensions(signal):
            raise ValueError("EEGNet input dimensions must be positive.")
        if signal.shape[1:] != self.input_shape[1:]:
            raise ValueError("EEGNet input channel or time sample count differs from the model input shape.")
        if not has_only_finite_numbers(signal):
            raise ValueError("EEGNet input contains NaN or infinity after float32 conversion.")

        if self._configuration.input_normalization == PER_EPOCH_CHANNEL_NORMALIZATION:
            signal = self._normalize_input(signal)

        return torch.tensor(signal, dtype=torch.float32)

    def _create_optimizer(self) -> Optimizer:
        optimizer_name = self._configuration.training.optimizer.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self._network.parameters(),
                lr=self._configuration.training.learning_rate,
            )

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                self._network.parameters(),
                lr=self._configuration.training.learning_rate,
            )

        raise ValueError(f"Unsupported optimizer: {self._configuration.training.optimizer}")

    def _fit_label_mapping(self, y: NDArray[np.generic]) -> None:
        self._classes = np.unique(y)

        if len(self._classes) != self._configuration.n_classes:
            raise ValueError(
                f"Config n_classes={self._configuration.n_classes}, "
                f"but training data has {len(self._classes)} classes: {self._classes}"
            )

        self._class_to_index = {class_label: class_index for class_index, class_label in enumerate(self._classes)}

    def _encode_labels(self, y: NDArray[np.generic]) -> NDArray[np.generic]:
        if self._class_to_index is None:
            raise ValueError("Label mapping is not initialized. Call fit first.")

        unknown_labels = set(np.unique(y)) - set(self._class_to_index.keys())
        if unknown_labels:
            raise ValueError(f"Labels {sorted(unknown_labels)} were not seen in training labels.")

        return np.array(
            [self._class_to_index[label] for label in y],
            dtype=np.int64,
        )

    def _decode_labels(self, y_indices: NDArray[np.generic]) -> NDArray[np.generic]:
        if self._classes is None:
            raise ValueError("Label mapping is not initialized. Call fit first.")

        decoded_labels: NDArray[np.generic] = self._classes[y_indices]
        return decoded_labels
