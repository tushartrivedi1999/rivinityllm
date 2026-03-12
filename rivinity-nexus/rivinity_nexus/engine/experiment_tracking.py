from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rivinity_nexus.config.settings import get_settings

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class ExperimentTracker:
    """MLflow-based experiment tracking facade with safe no-op fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = (
            self.settings.experiment_tracking_enabled
            and self.settings.experiment_tracking_backend == "mlflow"
            and mlflow is not None
        )

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=run_name):
            yield

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        safe_params = {
            str(k): ("|".join(v) if isinstance(v, (list, tuple)) else v)
            for k, v in params.items()
            if v is not None and isinstance(v, (str, int, float, bool, list, tuple))
        }
        if safe_params:
            mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        safe_metrics = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if not safe_metrics:
            return
        if step is None:
            mlflow.log_metrics(safe_metrics)
            return
        for k, v in safe_metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_dataset(self, dataset_uri: str) -> None:
        if not self.enabled:
            return
        mlflow.set_tag("dataset_uri", dataset_uri)
        dataset_path = Path(dataset_uri)
        if dataset_path.exists() and dataset_path.is_file():
            mlflow.log_artifact(str(dataset_path), artifact_path="datasets")

    def log_checkpoint(self, checkpoint_path: str) -> None:
        if not self.enabled:
            return
        path = Path(checkpoint_path)
        if path.exists():
            mlflow.log_artifacts(str(path), artifact_path="checkpoints")
