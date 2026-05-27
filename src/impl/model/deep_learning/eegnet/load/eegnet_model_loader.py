from pathlib import Path
from typing import Any

import torch

from src.pipeline.context.run_context import RunContext
from src.types.interfaces.model.model_loader import IModelLoader

STEP_NAME = "EEGNET_LOADER"

class EEGNetModelLoader(IModelLoader):
    def load(self, model_path: Path, run_ctx: RunContext) -> Any:
        log = run_ctx.logger.for_step(STEP_NAME)

        if not model_path.exists():
            msg = f"Model at path {model_path} does not exist. No model was loaded"
            log.exception(msg)
            raise FileNotFoundError(msg)

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )