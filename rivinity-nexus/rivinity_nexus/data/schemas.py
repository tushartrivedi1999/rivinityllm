from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from rivinity_nexus.engine.distributed_launcher import LauncherType
from rivinity_nexus.engine.training import TrainingMethod
from rivinity_nexus.models.entities import (
    DatasetFormat,
    DatasetStatus,
    JobStatus,
    ModelSourceType,
    ModelStatus,
    QueueJobState,
    QueueJobType,
    UserRole,
)


class SignupRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=255)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    created_at: datetime

    model_config = {"from_attributes": True}


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=128)


class ApiKeyCreateResponse(BaseModel):
    id: int
    name: str
    key: str
    created_at: datetime


class ApiKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class QueueJobSubmitRequest(BaseModel):
    job_type: QueueJobType
    payload: dict


class QueueJobResponse(BaseModel):
    id: int
    job_type: QueueJobType
    state: QueueJobState
    progress: int
    payload_json: str
    result_json: str | None
    error_message: str | None
    celery_task_id: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelUploadRequest(BaseModel):
    model_name: str = Field(min_length=2, max_length=255)
    version: str = Field(default="v1", min_length=1, max_length=64)
    parameter_count: int = Field(default=0, ge=0)
    architecture: str = Field(min_length=2, max_length=128)
    source_type: ModelSourceType
    source_uri: str


class ModelVersionRequest(BaseModel):
    parameter_count: int = Field(default=0, ge=0)
    architecture: str = Field(min_length=2, max_length=128)
    source_type: ModelSourceType
    source_uri: str


class ModelResponse(BaseModel):
    id: int
    model_name: str
    version: str
    parameter_count: int
    architecture: str
    source_type: ModelSourceType
    source_uri: str
    storage_uri: str
    status: ModelStatus
    owner_id: int
    upload_date: datetime

    model_config = {"from_attributes": True}


class DatasetUploadRequest(BaseModel):
    dataset_name: str = Field(min_length=2, max_length=255)
    version: str = Field(default="v1", min_length=1, max_length=64)
    format: DatasetFormat
    source_uri: str


class DatasetPreprocessRequest(BaseModel):
    shard_count: int = Field(default=4, ge=1, le=1024)


class DatasetResponse(BaseModel):
    id: int
    dataset_name: str
    version: str
    format: DatasetFormat
    source_uri: str
    storage_uri: str
    cache_key: str
    preprocessed_uri: str | None
    shard_count: int
    status: DatasetStatus
    owner_id: int
    upload_date: datetime

    model_config = {"from_attributes": True}


class DatasetShardResponse(BaseModel):
    shard_path: str
    rank: int
    world_size: int


class TrainingRuntimeConfigRequest(BaseModel):
    method: TrainingMethod = TrainingMethod.supervised_fine_tuning
    distributed: bool = True
    launcher: LauncherType = LauncherType.torchrun
    deepspeed_enabled: bool = True
    fsdp_enabled: bool = False
    gradient_checkpointing: bool = True
    mixed_precision: str = Field(default="bf16", pattern="^(bf16|fp16|fp32)$")
    learning_rate: float = Field(default=2e-4, gt=0)
    train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    qlora_4bit: bool = True
    qlora_quant_type: str = Field(default="nf4", pattern="^(nf4|fp4)$")
    qlora_double_quant: bool = True
    cpu_offload: bool = True


class TrainingRequest(BaseModel):
    model_id: int
    dataset_uri: str
    max_steps: int = Field(default=100, ge=1)
    runtime: TrainingRuntimeConfigRequest = Field(default_factory=TrainingRuntimeConfigRequest)


class TrainingJobResponse(BaseModel):
    id: int
    model_id: int
    dataset_uri: str
    max_steps: int
    task_id: str | None
    status: JobStatus
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class InferenceRequest(BaseModel):
    model_name: str
    prompt: str = Field(default="", min_length=0)
    prompts: list[str] | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    backend: str = Field(default="auto", pattern="^(auto|airllm|vllm)$")
    use_kv_cache: bool = True


class InferenceResponse(BaseModel):
    backend: str
    outputs: list[str]
    token_counts: list[int]


class GpuAvailabilityResponse(BaseModel):
    vendor: str
    gpu_type: str
    available: int
    region: str


class GpuScheduleRequest(BaseModel):
    required_gpus: int = Field(ge=1)
    required_vram_gb: int = Field(default=16, ge=1)
    strategy: str = Field(default="least_loaded", pattern="^(least_loaded|vram_aware|cost_aware)$")
    gpu_type: str | None = None


class GpuNodeAllocationResponse(BaseModel):
    vendor_name: str
    node_id: str
    node_ip: str
    gpu_type: str
    vram: int
    allocated_gpus: int
    estimated_cost_per_hour: float
    free_vram_gb_after: int


class GpuAllocationPlanResponse(BaseModel):
    strategy: str
    allocations: list[GpuNodeAllocationResponse]
    total_allocated_gpus: int
    estimated_total_cost_per_hour: float


class GpuVendorNodeRegisterRequest(BaseModel):
    vendor_name: str = Field(min_length=2, max_length=128)
    node_ip: str = Field(min_length=7, max_length=64)
    gpu_type: str = Field(min_length=2, max_length=64)
    vram: int = Field(ge=1)
    price_per_hour: float = Field(gt=0)
    total_gpus: int = Field(default=1, ge=1)
    available_gpus: int = Field(default=1, ge=0)


class GpuVendorNodeResponse(BaseModel):
    id: int
    vendor_name: str
    node_ip: str
    gpu_type: str
    vram: int
    price_per_hour: float
    total_gpus: int
    available_gpus: int


class MoERuntimeProfileResponse(BaseModel):
    total_params_trillion: float
    num_experts: int
    top_k_experts: int
    dense_fp16_memory_tb: float
    dense_quantized_memory_tb: float
    active_params_billion: float
    active_memory_gb: float
    estimated_compute_reduction_x: float
    expert_shards: dict[str, list[int]]
    kv_cache: dict
    dynamic_expert_loading: str
