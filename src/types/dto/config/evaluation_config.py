from pydantic import BaseModel

from src.types.dto.config.astageconfig import AStageConfig


class EvaluationConfig(AStageConfig):
    _target_class = "src.impl.evaluator.dummy_evaluator.DummyEvaluator"
    metrics: list[str]