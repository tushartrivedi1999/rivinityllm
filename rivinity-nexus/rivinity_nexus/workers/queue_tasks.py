import json
import time
from datetime import datetime

from rivinity_nexus.core.database import SessionLocal
from rivinity_nexus.core.redis_client import publish_job_progress
from rivinity_nexus.models.entities import QueueJob, QueueJobState
from rivinity_nexus.monitoring.metrics import record_queue_latency
from rivinity_nexus.workers.celery_app import celery_app


def _set_job_state(job_id: int, state: QueueJobState, progress: int, result: dict | None = None, error: str | None = None) -> None:
    db = SessionLocal()
    try:
        job = db.query(QueueJob).filter(QueueJob.id == job_id).first()
        if not job:
            return
        job.state = state
        job.progress = progress
        if state == QueueJobState.RUNNING and job.created_at is not None:
            latency = datetime.utcnow() - job.created_at
            record_queue_latency(job.job_type.value, latency.total_seconds())
        if result is not None:
            job.result_json = json.dumps(result)
        if error is not None:
            job.error_message = error
        db.commit()
        publish_job_progress(job_id=job_id, state=state.value, progress=progress, message=error)
    finally:
        db.close()


@celery_app.task(bind=True, name="jobs.training_job")
def training_job(self, job_id: int, payload: dict) -> dict:
    try:
        _set_job_state(job_id, QueueJobState.RUNNING, 10)
        self.update_state(state="RUNNING", meta={"progress": 10})
        time.sleep(0.1)

        _set_job_state(job_id, QueueJobState.RUNNING, 60)
        self.update_state(state="RUNNING", meta={"progress": 60})
        time.sleep(0.1)

        result = {"artifact": f"trained:{payload.get('model_id')}", "dataset": payload.get("dataset_uri")}
        _set_job_state(job_id, QueueJobState.COMPLETED, 100, result=result)
        return result
    except Exception as exc:
        _set_job_state(job_id, QueueJobState.FAILED, 100, error=str(exc))
        raise


@celery_app.task(bind=True, name="jobs.inference_job")
def inference_job(self, job_id: int, payload: dict) -> dict:
    try:
        _set_job_state(job_id, QueueJobState.RUNNING, 25)
        self.update_state(state="RUNNING", meta={"progress": 25})
        time.sleep(0.1)

        output = f"[generated] {str(payload.get('prompt', ''))[:120]}"
        result = {"output": output}
        _set_job_state(job_id, QueueJobState.COMPLETED, 100, result=result)
        return result
    except Exception as exc:
        _set_job_state(job_id, QueueJobState.FAILED, 100, error=str(exc))
        raise


@celery_app.task(bind=True, name="jobs.dataset_processing_job")
def dataset_processing_job(self, job_id: int, payload: dict) -> dict:
    try:
        _set_job_state(job_id, QueueJobState.RUNNING, 20)
        self.update_state(state="RUNNING", meta={"progress": 20})
        time.sleep(0.1)

        _set_job_state(job_id, QueueJobState.RUNNING, 80)
        self.update_state(state="RUNNING", meta={"progress": 80})
        time.sleep(0.1)

        result = {"dataset_uri": payload.get("dataset_uri"), "status": "processed"}
        _set_job_state(job_id, QueueJobState.COMPLETED, 100, result=result)
        return result
    except Exception as exc:
        _set_job_state(job_id, QueueJobState.FAILED, 100, error=str(exc))
        raise
