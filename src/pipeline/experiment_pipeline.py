from src.pipeline.context.run_context import RunContext
from src.pipeline.run_context_factory import RunContextFactory
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.preprocessing import IPreprocessing


class ExperimentPipeline:
    def __init__(
        self,
        data_loader: IDataLoader,
        preprocessing: IPreprocessing
    ) -> None:
        self._data_loader = data_loader
        self._run_context_factory = RunContextFactory()
        self._preprocessing = preprocessing

    def run(self, config: ExperimentConfig) -> None:

        run_ctx: RunContext = self._run_context_factory.create(config, "test", "experiment_pipeline")
        raw_data: RawDataDTO = self._data_loader.run(config.dataset, run_ctx)

        preprocessing_input: PreprocessingInputDTO = PreprocessingInputDTO(raw_data, config.preprocessing)
        self._preprocessing.run(preprocessing_input, run_ctx)
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
