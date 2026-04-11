import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ModelFactory:
    """
    Static factory class responsible for instantiating EEG classification pipelines.

    This factory centralizes the creation of various signal processing and
    classification chains, ranging from traditional machine learning (Riemannian
    geometry, CSP) to deep learning architectures (EEGNet via Braindecode).
    """
    @staticmethod
    def create(method_id: str, params: dict) -> Pipeline:
        """
        Instantiates a specific classification pipeline based on the method identifier.

        The method constructs a Scikit-learn Pipeline object, ensuring that all
        components (transformers and estimators) are properly initialized with
        the provided hyperparameters.

        Args:
            method_id (str): The unique identifier for the desired model pipeline
                (e.g., 'csp_lda', 'riemannian_svm', 'eegnet_braindecode').
            params (dict): A dictionary containing model-specific hyperparameters
                and metadata (e.g., n_channels, n_classes, learning_rate).

        Returns:
            Pipeline: A fully initialized Scikit-learn Pipeline ready for training.

        Raises:
            ValueError: If the provided 'method_id' does not match any implemented
                model architecture.
        """
        if method_id == "csp_lda":
            from mne.decoding import CSP
            return Pipeline([
                ("csp", CSP(n_components=params.get("n_components", 4))),
                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            ])

        if method_id == "riemannian_lda":
            return Pipeline(
                [("cov", Covariances(estimator=params.get("estimator", "oas"))),
                 ("ts", TangentSpace(metric=params.get("metric", "riemann"))),
                 ("scaler", StandardScaler()),
                 ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]
            )

        if method_id == "riemannian_svm":
            return Pipeline(
                [("cov", Covariances(estimator=params.get("estimator", "oas"))),
                 ("ts", TangentSpace(metric=params.get("metric", "riemann"))),
                 ("scaler", StandardScaler()),
                 ("svm", SVC(probability=True, kernel=params.get("kernel", "linear")))]
            )

        if method_id == "riemannian_mdm":
            from pyriemann.classification import MDM
            return Pipeline([
                ("cov", Covariances(estimator=params.get("estimator", "oas"))),
                ("mdm", MDM(metric=params.get("metric", "riemann")))
            ])

        if method_id == "riemannian_lr":
            from sklearn.linear_model import LogisticRegression
            return Pipeline([
                ("cov", Covariances(estimator=params.get("estimator", "oas"))),
                ("ts", TangentSpace(metric=params.get("metric", "riemann"))),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    penalty=params.get("penalty", "elasticnet"),
                    solver=params.get("solver", "saga"),
                    l1_ratio=params.get("l1_ratio", 0.5),
                    random_state=params.get("random_state", 42)
                ))
            ])

        if method_id == "riemannian_rf":
            from sklearn.ensemble import RandomForestClassifier
            return Pipeline([
                ("cov", Covariances(estimator=params.get("estimator", "oas"))),
                ("ts", TangentSpace(metric=params.get("metric", "riemann"))),
                ("rf", RandomForestClassifier(n_estimators=params.get("n_estimators", 100)))
            ])

        # Deep learning - EEGNet and Braindecode with Skorch wrapper

        #if method_id == "eegnet":
        #    net = NeuralNetClassifier(
        #        module=EEGNet,
        #        module__n_classes=params.get("n_classes", 2),
        #        module__n_channels=params.get("n_channels", 22),
        #        # skorch parameters (optimizer, loss, etc.)
        #        criterion=torch.nn.CrossEntropyLoss,
        #        optimizer=torch.optim.Adam,
        #        lr=params.get("lr", 0.001),
        #        max_epochs=params.get("epochs", 50),
        #        batch_size=params.get("batch_size", 32),
        #        device="cuda" if torch.cuda.is_available() else "cpu"
        #    )
        #
        #    return Pipeline([
        #        ("net", net)
        #    ])

        if method_id == "eegnet_braindecode":
            net = EEGClassifier(
                model=EEGNetv4,
                module__n_chans=params.get("n_channels", 22),
                module__n_outputs=params.get("n_classes", 2),
                module__input_window_samples=params.get("n_times", 1000),
                module__final_conv_length='auto',
                # Skorch parameters
                max_epochs=params.get("epochs", 50),
                batch_size=params.get("batch_size", 32),
                optimizer=torch.optim.Adam,
                lr=params.get("lr", 0.001),
                device="cuda" if torch.cuda.is_available() else "cpu",
                train_split=None
            )

            return Pipeline([
                #("reshape", Ensure4D()),
                ("net", net)
            ])

        raise ValueError(f"Method '{method_id}' is not yet implemented.")
