from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from rivinity_nexus.api.deps import get_current_user, get_db
from rivinity_nexus.data.schemas import (
    DatasetPreprocessRequest,
    DatasetResponse,
    DatasetShardResponse,
    DatasetUploadRequest,
)
from rivinity_nexus.engine.dataset_ingestion import DatasetIngestionService
from rivinity_nexus.models.entities import DatasetArtifact, User

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _get_dataset_or_404(db: Session, owner_id: int, dataset_id: int) -> DatasetArtifact:
    dataset = db.query(DatasetArtifact).filter(DatasetArtifact.id == dataset_id, DatasetArtifact.owner_id == owner_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.post("/upload", response_model=DatasetResponse, status_code=201)
def upload_dataset(
    payload: DatasetUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetResponse:
    return DatasetIngestionService(db).upload_dataset(
        owner_id=current_user.id,
        dataset_name=payload.dataset_name,
        version=payload.version,
        dataset_format=payload.format,
        source_uri=payload.source_uri,
    )


@router.get("", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[DatasetResponse]:
    return DatasetIngestionService(db).list_datasets(current_user.id)


@router.post("/{dataset_id}/preprocess", response_model=DatasetResponse)
def preprocess_dataset(
    dataset_id: int,
    payload: DatasetPreprocessRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetResponse:
    service = DatasetIngestionService(db)
    dataset = _get_dataset_or_404(db, current_user.id, dataset_id)
    return service.preprocess_and_shard(dataset, shard_count=payload.shard_count)


@router.get("/{dataset_id}/shard", response_model=DatasetShardResponse)
def get_distributed_shard(
    dataset_id: int,
    rank: int,
    world_size: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetShardResponse:
    service = DatasetIngestionService(db)
    dataset = _get_dataset_or_404(db, current_user.id, dataset_id)
    shard_path = service.get_distributed_shard_path(dataset, rank=rank, world_size=world_size)
    return DatasetShardResponse(shard_path=shard_path, rank=rank, world_size=world_size)
