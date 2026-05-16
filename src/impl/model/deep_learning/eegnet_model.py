import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer

from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.model.train_history import TrainingHistory
from src.types.interfaces.model.model import IModel


PER_EPOCH_CHANNEL_NORMALIZATION = "per_epoch_channel"

class EEGNetModel(IModel):
    """
    IModel wrapper around a PyTorch EEGNet network.
    """

    def __init__(self, network: nn.Module, model_name: str, config: EEGNetConfig) -> None:
        self._network = network
        self._model_name = model_name
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._network.to(self._device)
        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer: Optimizer = self._create_optimizer()

        self.best_epoch: int | None = None
        self.best_validation_accuracy: float | None = None
        self.history: TrainingHistory | None = None

        self._classes: np.ndarray | None = None
        self._class_to_index: dict | None = None

    def name(self) -> str:
        return self._model_name

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Basic fit required by IModel.

        Trains on one dataset for the configured number of epochs.
        No fold validation and no external evaluation is performed.
        """

        self.initialize_training(y)
        for _ in range(self._config.training.epochs):
            self.train_one_epoch(x, y)

    def initialize_training(self, y_train: np.ndarray) -> None:
        if self._classes is None:
            self._fit_label_mapping(y_train)

        self.history = TrainingHistory(
            train_loss=[],
            validation_loss=[],
            train_metrics={"accuracy": []},
            validation_metrics={
                "fold_accuracy": [],
                "validation_accuracy": [],
            },
        )

    def train_one_epoch(self, x_train: np.ndarray, y_train: np.ndarray) -> tuple[float, float]:
        if self.history is None:
            raise ValueError("Training is not initialized. Call initialize_training first.")

        y_train_encoded = self._encode_labels(y_train)
        train_loader = self._create_loader(x_train, y_train_encoded, shuffle=True)

        train_loss, train_accuracy = self._run_training_loader(train_loader)

        self.history.train_loss.append(train_loss)
        self.history.train_metrics["accuracy"].append(train_accuracy)

        return train_loss, train_accuracy

    def validate(self, x_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
        if self.history is None:
            raise ValueError("Training is not initialized. Call initialize_training first.")

        y_val_encoded = self._encode_labels(y_val)
        val_loader = self._create_loader(x=x_val, y=y_val_encoded, shuffle=False)

        val_loss, val_accuracy = self._run_evaluation_loader(val_loader)

        self.history.validation_loss.append(val_loss)
        self.history.validation_metrics["fold_accuracy"].append(val_accuracy)

        return val_loss, val_accuracy

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if self.history is None:
            raise ValueError("Training is not initialized. Call initialize_training first.")

        y_encoded = self._encode_labels(y)
        data_loader = self._create_loader(x, y_encoded, shuffle=False)

        loss, accuracy = self._run_evaluation_loader(data_loader)

        return loss, accuracy

    def _run_evaluation_loader(self, evaluation_loader: DataLoader) -> tuple[float, float]:
        self._network.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in evaluation_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                logits = self._network(batch_x)
                loss = self._loss_fn(logits, batch_y)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)

                total_correct += (predictions == batch_y).sum().item()
                total_samples += batch_y.size(0)

        return total_loss / len(evaluation_loader), total_correct / total_samples

    def _run_training_loader(self, train_loader: DataLoader) -> tuple[float, float]:
        self._network.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self._device)
            batch_y = batch_y.to(self._device)

            logits = self._network(batch_x)
            loss = self._loss_fn(logits, batch_y)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)

            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)

        return total_loss / len(train_loader), total_correct / total_samples

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_class_probability(x)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self._decode_labels(predicted_indices)

    def predict_class_probability(self, x: np.ndarray) -> np.ndarray:
        self._network.eval()

        x_tensor = self._to_x_tensor(x)
        data_loader = DataLoader(
            TensorDataset(x_tensor),
            batch_size=self._config.training.batch_size,
            shuffle=False,
        )

        probabilities_batches = []

        with torch.no_grad():
            for (batch_x,) in data_loader:
                batch_x = batch_x.to(self._device)

                logits = self._network(batch_x)
                probabilities = torch.softmax(logits, dim=1)

                probabilities_batches.append(probabilities.cpu().numpy())

        return np.concatenate(probabilities_batches, axis=0)

    def get_state_dict(self) -> dict:
        return self._network.state_dict()

    def get_network_state_dict(self) -> dict:
        return copy.deepcopy(self._network.state_dict())

    def load_network_state_dict(self, state_dict: dict) -> None:
        self._network.load_state_dict(state_dict)

    def _create_loader(
        self,
        x: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        x_tensor = self._to_x_tensor(x)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)

        return DataLoader(
            dataset,
            batch_size=self._config.training.batch_size,
            shuffle=shuffle,
        )

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=2, keepdims=True)
        std = x.std(axis=2, keepdims=True)

        return (x - mean) / (std + 1e-6)

    def _to_x_tensor(self, x: np.ndarray) -> torch.Tensor:
        x_array = np.asarray(x, dtype=np.float32)

        if x_array.ndim != 3:
            raise ValueError(
                f"EEGNet expects data shape "
                f"(n_epochs, n_channels, n_times), got {x_array.shape}"
            )

        if (self._config.input_normalization == PER_EPOCH_CHANNEL_NORMALIZATION):
            x_array = self._normalize_input(x_array)

        return torch.tensor(x_array, dtype=torch.float32)

    def _create_optimizer(self) -> Optimizer:
        optimizer_name = self._config.training.optimizer.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self._network.parameters(),
                lr=self._config.training.learning_rate,
            )

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                self._network.parameters(),
                lr=self._config.training.learning_rate,
            )

        raise ValueError(f"Unsupported optimizer: {self._config.training.optimizer}")

    def _fit_label_mapping(self, y: np.ndarray) -> None:
        self._classes = np.unique(y)

        if len(self._classes) != self._config.n_classes:
            raise ValueError(
                f"Config n_classes={self._config.n_classes}, "
                f"but training data has {len(self._classes)} classes: {self._classes}"
            )

        self._class_to_index = {
            class_label: index
            for index, class_label in enumerate(self._classes)
        }

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        if self._class_to_index is None:
            raise ValueError("Label mapping is not initialized. Call fit first.")

        unknown_labels = set(np.unique(y)) - set(self._class_to_index.keys())
        if unknown_labels:
            raise ValueError(
                f"Labels {sorted(unknown_labels)} were not seen in training labels."
            )

        return np.array(
            [self._class_to_index[label] for label in y],
            dtype=np.int64,
        )

    def _decode_labels(self, y_indices: np.ndarray) -> np.ndarray:
        if self._classes is None:
            raise ValueError("Label mapping is not initialized. Call fit first.")

        return self._classes[y_indices]

