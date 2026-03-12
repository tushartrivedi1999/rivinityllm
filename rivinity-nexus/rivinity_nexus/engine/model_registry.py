from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from rivinity_nexus.engine.storage import ModelStorageService
from rivinity_nexus.models.entities import ModelArtifact, ModelSourceType, ModelStatus


class ModelRegistryService:
    def __init__(self, db: Session):
        self.db = db
        self.storage = ModelStorageService()

    def register_model(
        self,
        model_name: str,
        version: str,
        parameter_count: int,
        architecture: str,
        source_type: ModelSourceType,
        source_uri: str,
        owner_id: int,
    ) -> ModelArtifact:
        existing = (
            self.db.query(ModelArtifact)
            .filter(
                ModelArtifact.owner_id == owner_id,
                ModelArtifact.model_name == model_name,
                ModelArtifact.version == version,
            )
            .first()
        )
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model version already exists")

        storage_uri = self.storage.store_reference(owner_id, model_name, version, source_uri)
        artifact = ModelArtifact(
            model_name=model_name,
            version=version,
            parameter_count=parameter_count,
            architecture=architecture,
            source_type=source_type,
            source_uri=source_uri,
            storage_uri=storage_uri,
            status=ModelStatus.uploaded,
            owner_id=owner_id,
        )
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def list_models(self, owner_id: int) -> list[ModelArtifact]:
        return (
            self.db.query(ModelArtifact)
            .filter(ModelArtifact.owner_id == owner_id)
            .order_by(ModelArtifact.model_name.asc(), ModelArtifact.upload_date.desc())
            .all()
        )

    def get_model(self, model_id: int, owner_id: int) -> ModelArtifact:
        model = self.db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.owner_id == owner_id).first()
        if not model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
        return model

    def create_next_version(
        self,
        owner_id: int,
        model_name: str,
        parameter_count: int,
        architecture: str,
        source_type: ModelSourceType,
        source_uri: str,
    ) -> ModelArtifact:
        latest = (
            self.db.query(ModelArtifact)
            .filter(ModelArtifact.owner_id == owner_id, ModelArtifact.model_name == model_name)
            .order_by(ModelArtifact.upload_date.desc())
            .first()
        )
        if latest and latest.version.startswith("v") and latest.version[1:].isdigit():
            version = f"v{int(latest.version[1:]) + 1}"
        else:
            version = "v1"
        return self.register_model(
            model_name=model_name,
            version=version,
            parameter_count=parameter_count,
            architecture=architecture,
            source_type=source_type,
            source_uri=source_uri,
            owner_id=owner_id,
        )
