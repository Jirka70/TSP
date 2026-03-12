from src.pipeline.context.run_context import RunContext
from src.pipeline.run_context_factory import RunContextFactory
from src.pipeline.stages.augmentation_stage import AugmentationStage
from src.pipeline.stages.config_validation_stage import ConfigValidationStage
from src.pipeline.stages.data_loading_stage import DataLoadingStage
from src.pipeline.stages.evaluation_stage import EvaluationStage
from src.pipeline.stages.model_training_stage import ModelTrainingStage
from src.pipeline.stages.preprocessing_stage import PreprocessingStage
from src.pipeline.stages.raw_data_validation_stage import RawDataValidationStage
from src.pipeline.stages.sample_preparation_stage import SamplePreparationStage
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.dto.training.sample_preparation_input_dto import SamplePreparationInputDTO
from src.types.dto.training.training_input_dto import TrainingInputDTO
from src.types.interfaces.data_loader import IDataLoader


class ExperimentPipeline:
    def __init__(
        self,
        data_loader: IDataLoader
    ) -> None:
        self._data_loader = data_loader
        self._run_context_factory = RunContextFactory()

    def run(self, config: ExperimentConfig) -> None:

        #run_ctx: RunContext = self._run_context_factory.create(config, "test", "experiment_pipeline")
        #self._data_loader.run(DatasetConfig(), run_ctx)
        """
        config_validation = self._config_validation_stage.run(config)
        if not config_validation.is_valid:
            raise ValueError(config_validation.messages)

        valid_config = config_validation.value
        assert valid_config is not None

        run_context = self._run_context_factory.create(valid_config)

        data_loading_input = DataLoadingInputDTO(
            dataset_name=valid_config.dataset.name,
            dataset_path=str(valid_config.dataset.path),
            subject_ids=valid_config.dataset.subject_ids,
            session_ids=valid_config.dataset.session_ids,
            run_ids=valid_config.dataset.run_ids,
            task=valid_config.dataset.task,
        )
        raw_data = self._data_loading_stage.run(data_loading_input)

        raw_data_validation = self._raw_data_validation_stage.run(raw_data)
        if not raw_data_validation.is_valid:
            raise ValueError(raw_data_validation.messages)

        valid_raw_data = raw_data_validation.value
        assert valid_raw_data is not None

        preprocessing_input = PreprocessingInputDTO(
            raw_data=valid_raw_data,
            l_freq=valid_config.preprocessing.l_freq,
            h_freq=valid_config.preprocessing.h_freq,
            notch_freq=valid_config.preprocessing.notch_freq,
            target_sfreq=valid_config.preprocessing.target_sfreq,
            rereference=valid_config.preprocessing.rereference,
            channel_selection=valid_config.preprocessing.channel_selection,
        )
        preprocessed_data = self._preprocessing_stage.run(preprocessing_input)

        sample_preparation_input = SamplePreparationInputDTO(
            preprocessed_data=preprocessed_data,
        )
        prepared_samples = self._sample_preparation_stage.run(sample_preparation_input)

        augmentation_input = AugmentationInputDTO(
            samples=prepared_samples,
            enabled=valid_config.augmentation.enabled,
        )
        augmented_samples = self._augmentation_stage.run(augmentation_input)

        training_input = TrainingInputDTO(
            samples=augmented_samples,
            n_classes=valid_config.model.n_classes,
        )
        trained_model = self._model_training_stage.run(training_input)

        evaluation_input = EvaluationInputDTO(
            trained_model=trained_model,
            samples=augmented_samples,
            metrics=valid_config.evaluation.metrics,
        )
        evaluation = self._evaluation_stage.run(evaluation_input, run_context)
        """
