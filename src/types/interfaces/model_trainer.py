from abc import ABC, abstractmethod


class IModelTrainer(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError
