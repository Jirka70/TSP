from types.interfaces.model.model import IModel


import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.types.dto.config.model.model_config import ModelConfig
from src.types.dto.model.train_history import TrainingHistory
from src.types.interfaces.model.model import IModel


class EEGNetModel(IModel):
    """
    IModel wrapper around a PyTorch EEGNet network.
    """

    def __init__(self, network: nn.Module, model_name: str, config: ModelConfig) -> None:
        self._network = network
        self._model_name = model_name
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._network.to(self._device)

        self.best_epoch: int | None = None
        self.best_validation_accuracy: float | None = None
        self.history: TrainingHistory | None = None

    def name(self) -> str:
        return self._model_name

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Basic fit required by IModel.

        This trains without validation data. For DL training with validation,
        use fit_with_validation().
        """
        self.history = self.fit_with_validation(
            x_train=x,
            y_train=y,
            x_val=None,
            y_val=None,
        )

    def fit_with_validation(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainingHistory:
        train_loader = self._create_loader(x_train, y_train, shuffle=True)

        val_loader = None
        if x_val is not None and y_val is not None:
            val_loader = self._create_loader(x_val, y_val, shuffle=False)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = self._create_optimizer()

        history = TrainingHistory(
            train_loss=[],
            validation_loss=[],
            train_metrics={"accuracy": []},
            validation_metrics={"accuracy": []},
        )

        best_state = None
        best_val_accuracy = -1.0

        for epoch in range(self._config.training.epochs):
            train_loss, train_accuracy = self._train_one_epoch(
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )

            history.train_loss.append(train_loss)
            history.train_metrics["accuracy"].append(train_accuracy)

            if val_loader is not None:
                val_loss, val_accuracy = self._evaluate(
                    data_loader=val_loader,
                    loss_fn=loss_fn,
                )

                history.validation_loss.append(val_loss)
                history.validation_metrics["accuracy"].append(val_accuracy)

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_state = copy.deepcopy(self._network.state_dict())
                    self.best_epoch = epoch
                    self.best_validation_accuracy = val_accuracy

        if best_state is not None:
            self._network.load_state_dict(best_state)

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_class_probability(x)
        return np.argmax(probabilities, axis=1)

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
        return {
            "model_name": self._model_name,
            "network_state_dict": self._network.state_dict(),
            "config": self._config.model_dump(),
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_validation_accuracy,
        }

    def _train_one_epoch(self, train_loader, loss_fn, optimizer) -> tuple[float, float]:
        self._network.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self._device)
            batch_y = batch_y.to(self._device)

            logits = self._network(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)

            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)

        return total_loss / len(train_loader), total_correct / total_samples

    def _evaluate(self, data_loader, loss_fn) -> tuple[float, float]:
        self._network.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                logits = self._network(batch_x)
                loss = loss_fn(logits, batch_y)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)

                total_correct += (predictions == batch_y).sum().item()
                total_samples += batch_y.size(0)

        return total_loss / len(data_loader), total_correct / total_samples

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

    def _to_x_tensor(self, x: np.ndarray) -> torch.Tensor:
        x_array = np.asarray(x, dtype=np.float32)

        if x_array.ndim != 3:
            raise ValueError(
                f"EEGNet expects data shape "
                f"(n_epochs, n_channels, n_times), got {x_array.shape}"
            )

        return torch.tensor(x_array, dtype=torch.float32)

    def _create_optimizer(self):
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
