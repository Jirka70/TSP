from src.types.interfaces.model.model import IModel


class DummyModel(IModel):
    def name(self) -> str:
        return ""

    def fit(self, x, y) -> None:
        pass

    def predict(self, x):
        pass

    def predict_class_probability(self, x):
        pass
