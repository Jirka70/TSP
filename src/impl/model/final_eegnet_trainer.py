import logging

import numpy as np

from src.impl.model.eegnet_model import EEGNetModel
from src.impl.model.util.extract.extract_learning_data import extract_learning_data
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.interfaces.model.final_trainer import IFinalTrainer

log = logging.getLogger(__name__)


class FinalEEGNetTrainer(IFinalTrainer):
    """
    Trains one final EEGNet model.

    This trainer is not for fold-based evaluation. Prefer passing train_data that
    contains each final-training sample once. Cross-validation folds are accepted
    only as a backward-compatible fallback.
    """

    def run(
        self,
        input_dto: FinalTrainingInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[FinalTrainingResultDTO]:
        network = create_eegnet_network(input_dto.config)
        model = EEGNetModel(network=network,
                            model_name=input_dto.config.model_name,
                            config=input_dto.config)

        all_y_train = np.concatenate([
            extract_learning_data(fold.train_data)[1]
            for fold in input_dto.folds
        ])

        return None

