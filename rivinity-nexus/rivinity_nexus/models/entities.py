from datetime import datetime
from enum import Enum

from sqlalchemy import BigInteger, Boolean, DateTime, Enum as SqlEnum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rivinity_nexus.core.database import Base


class ModelStatus(str, Enum):
    uploaded = "uploaded"
    training = "training"
    ready = "ready"
    failed = "failed"


class ModelSourceType(str, Enum):
    huggingface = "huggingface"
    local_checkpoint = "local_checkpoint"


class DatasetFormat(str, Enum):
    jsonl = "jsonl"
    parquet = "parquet"
    huggingface = "huggingface"


class DatasetStatus(str, Enum):
    uploaded = "uploaded"
    preprocessed = "preprocessed"
    failed = "failed"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class QueueJobType(str, Enum):
    training_job = "training_job"
    inference_job = "inference_job"
    dataset_processing_job = "dataset_processing_job"


class QueueJobState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class UserRole(str, Enum):
    admin = "admin"
    user = "user"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(SqlEnum(UserRole), default=UserRole.user)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    models: Mapped[list["ModelArtifact"]] = relationship(back_populates="owner")
    datasets: Mapped[list["DatasetArtifact"]] = relationship(back_populates="owner")
    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="user")
    sessions: Mapped[list["Session"]] = relationship(back_populates="user")
    queue_jobs: Mapped[list["QueueJob"]] = relationship(back_populates="user")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    key_prefix: Mapped[str] = mapped_column(String(12), index=True)
    hashed_key: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="api_keys")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    refresh_token_hash: Mapped[str] = mapped_column(String(255), unique=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="sessions")


class QueueJob(Base):
    __tablename__ = "queue_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    job_type: Mapped[QueueJobType] = mapped_column(SqlEnum(QueueJobType), index=True)
    state: Mapped[QueueJobState] = mapped_column(SqlEnum(QueueJobState), default=QueueJobState.PENDING, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    payload_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="queue_jobs")


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (UniqueConstraint("owner_id", "model_name", "version", name="uq_model_owner_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    parameter_count: Mapped[int] = mapped_column(BigInteger, default=0)
    architecture: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[ModelSourceType] = mapped_column(SqlEnum(ModelSourceType), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[ModelStatus] = mapped_column(SqlEnum(ModelStatus), default=ModelStatus.uploaded)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    upload_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    owner: Mapped[User] = relationship(back_populates="models")
    jobs: Mapped[list["TrainingJob"]] = relationship(back_populates="model")


class DatasetArtifact(Base):
    __tablename__ = "dataset_artifacts"
    __table_args__ = (UniqueConstraint("owner_id", "dataset_name", "version", name="uq_dataset_owner_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    format: Mapped[DatasetFormat] = mapped_column(SqlEnum(DatasetFormat), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(128), index=True)
    preprocessed_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    shard_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[DatasetStatus] = mapped_column(SqlEnum(DatasetStatus), default=DatasetStatus.uploaded)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    upload_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    owner: Mapped[User] = relationship(back_populates="datasets")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_artifacts.id"), index=True)
    dataset_uri: Mapped[str] = mapped_column(Text, nullable=False)
    max_steps: Mapped[int] = mapped_column(Integer, default=100)
    task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[JobStatus] = mapped_column(SqlEnum(JobStatus), default=JobStatus.queued)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped[ModelArtifact] = relationship(back_populates="jobs")


class GpuVendorNode(Base):
    __tablename__ = "gpu_vendor_nodes"
    __table_args__ = (UniqueConstraint("node_ip", name="uq_gpu_vendor_node_ip"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    vendor_name: Mapped[str] = mapped_column(String(128), index=True)
    node_ip: Mapped[str] = mapped_column(String(64), nullable=False)
    gpu_type: Mapped[str] = mapped_column(String(64), index=True)
    vram: Mapped[int] = mapped_column(Integer)
    price_per_hour: Mapped[float] = mapped_column(Float)
    total_gpus: Mapped[int] = mapped_column(Integer, default=1)
    available_gpus: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
