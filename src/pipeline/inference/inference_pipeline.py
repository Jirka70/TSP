from src.pipeline.context.run_context import RunContext
from src.pipeline.pipeline import IPipeline
from src.types.dto.config.experiment_config import ExperimentConfig


class InferencePipeline(IPipeline):
    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        pass
