import logging
from pathlib import Path

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.pipeline import IPipeline
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO # Předpokládaný import
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.artifact_saver import IArtifactSaver
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from src.types.interfaces.evaluator import IEvaluator
from src.types.interfaces.paradigm import IParadigm
from src.types.interfaces.raw_preprocessing import IRawPreprocessing
from src.types.interfaces.visualizer import IVisualizer
from src.types.interfaces.model.model_loader import IModelLoader # Tvůj nový interface
from src.types.interfaces.model.model_serializer import IModelSerializer


class ExperimentPipeline(IPipeline):
    def __init__(
            self,
            data_loader: IDataLoader,
            raw_preprocessing: IRawPreprocessing,
            paradigm: IParadigm,
            epoch_preprocessing: IEpochPreprocessing,
            model_loader: IModelLoader, # Přidáno
            evaluator: IEvaluator,
            visualizer: IVisualizer,
            artifact_saver: IArtifactSaver,
            model_serializer: IModelSerializer,
    ) -> None:
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
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.source, run_ctx)

        raw_preprocessing_input = RawPreprocessingInputDTO(config.raw_preprocessing, load_result.data)
        raw_preprocessing_result: StepResult[RawPreprocessedDTO] = self._raw_preprocessing.run(raw_preprocessing_input, run_ctx)
        self._visualizer.visualize_raw(raw_preprocessing_result.data, run_ctx)

        paradigm_input = ParadigmInputDTO(config.paradigm, raw_preprocessing_result.data)
        paradigm_result: StepResult[ParadigmResultDTO] = self._paradigm.run(paradigm_input, run_ctx)

        epoch_preprocessing_input = EpochPreprocessingInputDTO(config.epoch_preprocessing, paradigm_result.data)
        epoch_preprocessing_result: StepResult[EpochPreprocessedDTO] = self._epoch_preprocessing.run(epoch_preprocessing_input, run_ctx)
        self._visualizer.visualize_epochs(epoch_preprocessing_result.data, run_ctx)

        model_path = Path(config.model_path.path)
        loaded_model_obj = self._model_loader.load(model_path)

        trained_model = TrainedModelDTO(
            model=loaded_model_obj,
            model_name=model_path.stem
        )

        dummy_split = DatasetSplitDTO(
            folds=[],
            validation_data=epoch_preprocessing_result.data
        )

        evaluation_input = EvaluationInputDTO(
            config=config.evaluation,
            trained_models=[trained_model],
            folds=[],
            dataset_split=dummy_split
        )
        evaluation_result = self._evaluator.run(evaluation_input, run_ctx)

        self._visualizer.visualize_evaluation(
            evaluation_result.data,
            run_ctx,
            trained_model.model_name
        )

        save_artifacts_input = SaveArtifactsInputDTO(
            config.save_artifacts,
            config,
            output_path=Path("experiment_results.txt"),
            evaluation_result=evaluation_result.data,
        )

        self._artifact_saver.run(save_artifacts_input, run_ctx)
        self._log.info("Experiment pipeline finished successfully.")