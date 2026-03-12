from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from rivinity_nexus.api.deps import get_current_user, get_db
from rivinity_nexus.data.schemas import ModelResponse, ModelUploadRequest, ModelVersionRequest
from rivinity_nexus.engine.model_registry import ModelRegistryService
from rivinity_nexus.models.entities import User

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/upload", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def upload_model(
    payload: ModelUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelResponse:
    return ModelRegistryService(db).register_model(
        model_name=payload.model_name,
        version=payload.version,
        parameter_count=payload.parameter_count,
        architecture=payload.architecture,
        source_type=payload.source_type,
        source_uri=payload.source_uri,
        owner_id=current_user.id,
    )


@router.get("", response_model=list[ModelResponse])
def list_models(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[ModelResponse]:
    return ModelRegistryService(db).list_models(owner_id=current_user.id)


@router.post("/{model_name}/versions", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def create_model_version(
    model_name: str,
    payload: ModelVersionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelResponse:
    return ModelRegistryService(db).create_next_version(
        owner_id=current_user.id,
        model_name=model_name,
        parameter_count=payload.parameter_count,
        architecture=payload.architecture,
        source_type=payload.source_type,
        source_uri=payload.source_uri,
    )
