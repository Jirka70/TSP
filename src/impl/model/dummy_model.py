from src.types.interfaces.model.model import IModel


class DummyModel(IModel):
    def name(self) -> str:
        pass

    def fit(self, x, y) -> None:
        pass

    def predict(self, x):
        pass