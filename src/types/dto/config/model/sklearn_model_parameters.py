from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SklearnModelParameters(BaseModel):
    """Common base class for sklearn model parameter DTOs."""

    model_config = ConfigDict(extra="forbid")


class CspLdaParameters(SklearnModelParameters):
    """Parameters for the CSP + LDA pipeline."""

    n_components: int = Field(default=4, ge=1)
    reg: str = Field(default="ledoit_wolf", min_length=1)


class RiemannianBaseParameters(SklearnModelParameters):
    """Shared parameters for all Riemannian-based pipelines."""

    estimator: str = Field(default="oas", min_length=1)
    metric: str = Field(default="riemann", min_length=1)


class RiemannianLdaParameters(RiemannianBaseParameters):
    """Parameters for the Riemannian + LDA pipeline."""

    pass


class RiemannianSvmParameters(RiemannianBaseParameters):
    """Parameters for the Riemannian + SVM pipeline."""

    kernel: str = Field(default="linear", min_length=1)


class RiemannianMdmParameters(RiemannianBaseParameters):
    """Parameters for the Riemannian + MDM pipeline."""

    pass


class RiemannianLrParameters(RiemannianBaseParameters):
    """Parameters for the Riemannian + Logistic Regression pipeline."""

    penalty: str = Field(default="elasticnet", min_length=1)
    solver: str = Field(default="saga", min_length=1)
    l1_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    random_state: int = Field(default=42)



class RiemannianRfParameters(RiemannianBaseParameters):
    """Parameters for the Riemannian + Random Forest pipeline."""

    n_estimators: int = Field(default=100, ge=1)


SKLEARN_MODEL_PARAMETER_MODELS: dict[str, type[SklearnModelParameters]] = {
    "csp_lda": CspLdaParameters,
    "riemannian_lda": RiemannianLdaParameters,
    "riemannian_svm": RiemannianSvmParameters,
    "riemannian_mdm": RiemannianMdmParameters,
    "riemannian_lr": RiemannianLrParameters,
    "riemannian_rf": RiemannianRfParameters,
}


def validate_sklearn_model_parameters(model_name: str | None, raw_parameters: Any) -> dict[str, Any]:
    """Validate and normalize sklearn model parameters for a specific model name."""
    if model_name is None:
        raise ValueError("Model name must be set before validating parameters.")

    params_model = SKLEARN_MODEL_PARAMETER_MODELS.get(model_name)
    if params_model is None:
        raise ValueError(
            f"Unsupported sklearn model '{model_name}'. Supported models are: {sorted(SKLEARN_MODEL_PARAMETER_MODELS)}"
        )

    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, Mapping):
        raise TypeError("parameters must be provided as a mapping/dictionary.")

    raw_parameters = dict(raw_parameters)

    validated = params_model.model_validate(raw_parameters)
    return validated.model_dump(exclude_none=True)

