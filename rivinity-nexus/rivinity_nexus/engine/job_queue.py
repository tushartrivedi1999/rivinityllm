import json
import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from rivinity_nexus.core.resilience import retry_call
from rivinity_nexus.models.entities import QueueJob, QueueJobState, QueueJobType
from rivinity_nexus.workers.queue_tasks import dataset_processing_job, inference_job, training_job

logger = logging.getLogger("rivinity_nexus.job_queue")


class DistributedJobQueueService:
    def __init__(self, db: Session):
        self.db = db

    def enqueue(self, user_id: int, job_type: QueueJobType, payload: dict) -> QueueJob:
        job = QueueJob(
            user_id=user_id,
            job_type=job_type,
            state=QueueJobState.PENDING,
            progress=0,
            payload_json=json.dumps(payload),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        task_map = {
            QueueJobType.training_job: training_job,
            QueueJobType.inference_job: inference_job,
            QueueJobType.dataset_processing_job: dataset_processing_job,
        }
        try:
            task = retry_call(
                lambda: task_map[job_type].delay(job_id=job.id, payload=payload),
                attempts=3,
                delay_seconds=0.2,
                op_name=f"enqueue_{job_type.value}",
            )
            job.celery_task_id = task.id
            self.db.commit()
            self.db.refresh(job)
        except Exception as exc:
            logger.exception("queue_dispatch_failed", exc_info=exc)
            job.state = QueueJobState.FAILED
            job.error_message = str(exc)
            self.db.commit()
            self.db.refresh(job)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Job queue temporarily unavailable") from exc
        return job

    def get(self, job_id: int, user_id: int) -> QueueJob:
        job = self.db.query(QueueJob).filter(QueueJob.id == job_id, QueueJob.user_id == user_id).first()
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return job

    def list(self, user_id: int) -> list[QueueJob]:
        return self.db.query(QueueJob).filter(QueueJob.user_id == user_id).order_by(QueueJob.created_at.desc()).all()
