import logging
from pathlib import Path
from typing import Any

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.pipeline.pipeline import IPipeline
from tsp_eeg_classification.types.dto.config.experiment_config import ExperimentConfig
from tsp_eeg_classification.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from tsp_eeg_classification.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from tsp_eeg_classification.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from tsp_eeg_classification.types.dto.load.raw_data_dto import RawDataDTO
from tsp_eeg_classification.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from tsp_eeg_classification.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from tsp_eeg_classification.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from tsp_eeg_classification.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO
from tsp_eeg_classification.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from tsp_eeg_classification.types.dto.model.trained_model_dto import TrainedModelDTO
from tsp_eeg_classification.types.dto.split.dataset_split_dto import DatasetSplitDTO
from tsp_eeg_classification.types.interfaces.artifact_saver import IArtifactSaver
from tsp_eeg_classification.types.interfaces.data_loader import IDataLoader
from tsp_eeg_classification.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from tsp_eeg_classification.types.interfaces.evaluator import IEvaluator
from tsp_eeg_classification.types.interfaces.paradigm import IParadigm
from tsp_eeg_classification.types.interfaces.raw_preprocessing import IRawPreprocessing
from tsp_eeg_classification.types.interfaces.visualizer import IVisualizer
from tsp_eeg_classification.types.interfaces.model.model_loader import IModelLoader
from tsp_eeg_classification.types.interfaces.model.model_serializer import IModelSerializer
from tsp_eeg_classification.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO


class ExperimentPipeline(IPipeline):
    """
    Orchestrator pipeline executing the end-to-end BCI evaluation workflow.

    Coordinates data loading, multi-stage preprocessing, model restoration,
    downstream classification evaluation, and experimental artifact persistence.
    """
    def __init__(
            self,
            data_loader: IDataLoader,
            raw_preprocessing: IRawPreprocessing,
            paradigm: IParadigm,
            epoch_preprocessing: IEpochPreprocessing,
            model_loader: IModelLoader,
            evaluator: IEvaluator,
            visualizer: IVisualizer,
            artifact_saver: IArtifactSaver,
            model_serializer: IModelSerializer,
    ) -> None:
        """Initialize the pipeline with all necessary functional stage engines."""
        self._log = logging.getLogger(__name__)
        self._data_loader = data_loader
        self._raw_preprocessing = raw_preprocessing
        self._paradigm = paradigm
        self._epoch_preprocessing = epoch_preprocessing
        self._model_loader = model_loader
        self._evaluator = evaluator
        self._visualizer = visualizer
        self._artifact_saver = artifact_saver
        self._model_serializer = model_serializer

    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        """
        Execute the configured evaluation experiment pipeline sequentially.

        Args:
            config (ExperimentConfig): The master configuration object for the experiment run.
            run_ctx (RunContext): Context keeping track of unique runtime metadata.
        """
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.source, run_ctx)


        raw_preprocessing_input: RawPreprocessingInputDTO = RawPreprocessingInputDTO(config.raw_preprocessing, load_result.data)
        raw_preprocessing_result: StepResult[RawPreprocessedDTO] = self._raw_preprocessing.run(raw_preprocessing_input, run_ctx)
        self._visualizer.visualize_raw(raw_preprocessing_result.data, run_ctx)

        paradigm_input: ParadigmInputDTO = ParadigmInputDTO(config.paradigm, raw_preprocessing_result.data)
        paradigm_result: StepResult[ParadigmResultDTO] = self._paradigm.run(paradigm_input, run_ctx)

        epoch_preprocessing_input: EpochPreprocessingInputDTO = EpochPreprocessingInputDTO(config.epoch_preprocessing, paradigm_result.data)
        epoch_preprocessing_result: StepResult[EpochPreprocessedDTO] = self._epoch_preprocessing.run(epoch_preprocessing_input, run_ctx)
        self._visualizer.visualize_epochs(epoch_preprocessing_result.data, run_ctx)

        model_path: Path = Path(config.model_path.path)
        loaded_model_obj: Any = self._model_loader.load(model_path)

        trained_model : TrainedModelDTO = TrainedModelDTO(
            model=loaded_model_obj,
            model_name=model_path.stem
        )

        dummy_split : DatasetSplitDTO = DatasetSplitDTO(
            folds=[],
            validation_data=epoch_preprocessing_result.data
        )

        evaluation_input : EvaluationInputDTO = EvaluationInputDTO(
            config=config.evaluation,
            trained_models=[trained_model],
            folds=[],
            dataset_split=dummy_split
        )
        evaluation_result : StepResult[EvaluationResultDTO] = self._evaluator.run(evaluation_input, run_ctx)

        self._visualizer.visualize_evaluation(
            evaluation_result.data,
            run_ctx,
            trained_model.model_name
        )

        save_artifacts_input : SaveArtifactsInputDTO = SaveArtifactsInputDTO(
            config.save_artifacts,
            config,
            output_path=Path("experiment_results.txt"),
            evaluation_result=evaluation_result.data,
        )

        # TODO: Implement custom artifact saver for experiment pipeline.
        #self._artifact_saver.run(save_artifacts_input, run_ctx)
        self._log.info("Experiment pipeline finished successfully.")