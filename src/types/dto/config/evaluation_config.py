from pydantic import BaseModel


class EvaluationConfig(BaseModel):
    metrics: list[str]