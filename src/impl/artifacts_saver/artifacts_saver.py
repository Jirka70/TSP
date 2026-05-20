import logging
import re
from pathlib import Path

from src.impl.model.util.serialize.write_json import write_json
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.save_artifacts.artifact_ref import ArtifactRef
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.save_artifacts.saved_artifacts_dto import SavedArtifactsDTO
from src.types.interfaces.artifact_saver import IArtifactSaver


class UnsupportedModelSerializerError(Exception):
    pass


log = logging.getLogger(__name__)


class ArtifactSaver(IArtifactSaver):
    def run(
            self, input_dto: SaveArtifactsInputDTO, run_ctx: RunContext
    ) -> StepResult[SavedArtifactsDTO]:
        saved_artifacts: list[ArtifactRef] = []

        output_path = self._create_artifact_output_path(input_dto, run_ctx)
        output_path.mkdir(parents=True, exist_ok=True)

        config = input_dto.config

        if config.save_model:
            model_artifacts = self._save_model(input_dto, output_path)
            saved_artifacts.extend(model_artifacts.artifacts)

        if config.save_metrics:
            metrics_artifact = self._save_metrics(input_dto, output_path)
            if metrics_artifact is not None:
                saved_artifacts.append(metrics_artifact)

        if config.save_config:
            config_artifact = self._save_config(input_dto, output_path)
            saved_artifacts.append(config_artifact)

        if config.save_training_history:
            history_artifact = self._save_training_history(input_dto, output_path)
            if history_artifact is not None:
                saved_artifacts.append(history_artifact)

        manifest_artifact = self._save_manifest(saved_artifacts, output_path, run_ctx)
        saved_artifacts.append(manifest_artifact)

        return StepResult(SavedArtifactsDTO(artifacts=saved_artifacts))

    def _create_artifact_output_path(
            self,
            input_dto: SaveArtifactsInputDTO,
            run_ctx: RunContext,
    ) -> Path:
        timestamp = run_ctx.started_at.strftime("%Y%m%d_%H%M%S")
        folder_name = self._build_artifact_folder_name(input_dto, timestamp)
        return input_dto.output_path / folder_name

    def _build_artifact_folder_name(
            self,
            input_dto: SaveArtifactsInputDTO,
            timestamp: str,
    ) -> str:
        model_name = (
            input_dto.trained_model.model_name
            if input_dto.trained_model is not None
            else "artifacts"
        )
        safe_model_name = self._to_safe_folder_name(model_name)
        return f"{safe_model_name}-{timestamp}"

    def _to_safe_folder_name(self, value: str) -> str:
        safe_value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
        return safe_value.strip("_") or "model"

    def _save_model(
            self,
            input_dto: SaveArtifactsInputDTO,
            output_path: Path,
    ) -> SavedArtifactsDTO:
        if input_dto.model_serializer is None:
            raise ValueError("Model serializer is required when save_model=True.")

        if input_dto.trained_model is None:
            raise ValueError("Trained model is required when save_model=True.")

        serializer = input_dto.model_serializer
        trained_model = input_dto.trained_model

        if not serializer.supports(trained_model.model_name):
            raise ValueError(
                f"Serializer {serializer.__class__.__name__} does not support "
                f"model {trained_model.model_name}."
            )

        return serializer.save(
            trained_model=trained_model,
            output_path=output_path / "models"
        )

    def _save_metrics(self, input_dto: SaveArtifactsInputDTO, output_path: Path) -> ArtifactRef | None:
        if input_dto.evaluation_result is None:
            log.warning("Evaluation result is missing. Metrics will not be saved.")
            return None

        file_path = output_path / "metrics.json"
        write_json(file_path, input_dto.evaluation_result)

        return ArtifactRef(
            name="metrics",
            path=file_path,
            kind="metrics",
            metadata=None,
        )

    def _save_config(self, input_dto: SaveArtifactsInputDTO, output_path: Path) -> ArtifactRef:
        file_path = output_path / "config.json"

        write_json(file_path, input_dto.experiment_config)

        return ArtifactRef(
            name="config_snapshot",
            path=file_path,
            kind="config",
            metadata=None
        )

    def _save_training_history(self, input_dto: SaveArtifactsInputDTO, output_path: Path) -> ArtifactRef | None:
        if input_dto.trained_model is None:
            log.warning("Trained model is missing. Training history will not be saved.")
            return None

        if input_dto.trained_model.history is None:
            log.warning("Training history is missing. Training history will not be saved.")
            return None

        file_path = output_path / "training_history.json"
        write_json(file_path, input_dto.trained_model.history)

        return ArtifactRef(
            name="training_history",
            path=file_path,
            kind="history",
            metadata={
                "model_name": input_dto.trained_model.model_name,
            },
        )

    def _save_manifest(self, artifacts: list[ArtifactRef], output_path: Path, run_ctx: RunContext) -> ArtifactRef:
        file_path = output_path / "manifest.json"

        manifest = {
            "run_id": run_ctx.run_id,
            "experiment_name": run_ctx.experiment_name,
            "pipeline_name": run_ctx.pipeline_name,
            "started_at": run_ctx.started_at.isoformat(),
            "git_commit_hash": run_ctx.git_commit_hash,
            "artifacts": artifacts,
        }

        write_json(file_path, manifest)

        return ArtifactRef(
            name="manifest",
            path=file_path,
            kind="manifest",
            metadata=None,
        )
