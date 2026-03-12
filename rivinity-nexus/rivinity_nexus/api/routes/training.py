from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from rivinity_nexus.api.deps import get_current_user, get_db
from rivinity_nexus.data.schemas import TrainingJobResponse, TrainingRequest
from rivinity_nexus.engine.model_registry import ModelRegistryService
from rivinity_nexus.engine.training import TrainingJobService
from rivinity_nexus.models.entities import User

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/start", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_training(
    payload: TrainingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrainingJobResponse:
    model = ModelRegistryService(db).get_model(payload.model_id, owner_id=current_user.id)
    try:
        return TrainingJobService(db).enqueue_training(
            model.id,
            payload.dataset_uri,
            payload.max_steps,
            config_overrides=payload.runtime.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Training worker unavailable") from exc
