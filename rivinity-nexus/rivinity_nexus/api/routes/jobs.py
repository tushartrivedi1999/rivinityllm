from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from rivinity_nexus.api.deps import get_current_user, get_db
from rivinity_nexus.data.schemas import QueueJobResponse, QueueJobSubmitRequest
from rivinity_nexus.engine.job_queue import DistributedJobQueueService
from rivinity_nexus.models.entities import User

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/submit", response_model=QueueJobResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_job(
    payload: QueueJobSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QueueJobResponse:
    return DistributedJobQueueService(db).enqueue(user_id=current_user.id, job_type=payload.job_type, payload=payload.payload)


@router.get("/{job_id}", response_model=QueueJobResponse)
def get_job(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> QueueJobResponse:
    return DistributedJobQueueService(db).get(job_id=job_id, user_id=current_user.id)


@router.get("", response_model=list[QueueJobResponse])
def list_jobs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[QueueJobResponse]:
    return DistributedJobQueueService(db).list(user_id=current_user.id)
