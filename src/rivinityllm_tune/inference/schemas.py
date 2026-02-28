"""Schemas for memory-aware inference."""

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    model_id: str
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=8192)
    memory_budget_gb: float = Field(default=8.0, gt=0)


class InferenceResponse(BaseModel):
    output: str
    strategy: str


class InferencePlanRequest(BaseModel):
    parameter_count_b: float = Field(gt=0, description="Model size in billions of parameters")
    quantization_bits: int = Field(default=4, description="Weight quantization bits")
    target_gpu_vram_gb: float = Field(default=8.0, gt=0)
    sequence_length: int = Field(default=4096, ge=128)
    batch_size: int = Field(default=1, ge=1)


class InferencePlanResponse(BaseModel):
    parameter_count_b: float
    quantization_bits: int
    weight_size_gb: float
    target_gpu_vram_gb: float
    gpu_count: int
    tensor_parallelism: int
    pipeline_parallelism: int
    cpu_cache_gb: float
    nvme_offload_gb: float
    prefetch_window_layers: int
    min_nvme_read_gbps: float
    strategy: str
