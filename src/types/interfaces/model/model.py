from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def predict_class_probability(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self) -> object:
        raise NotImplementedError
