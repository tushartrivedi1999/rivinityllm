# Rivinity Nexus FastAPI Server

Production-focused backend API for model upload, dataset ingestion, training orchestration, inference, GPU allocation, and distributed job processing.

## Stack

- FastAPI
- Pydantic v2
- SQLAlchemy ORM
- PostgreSQL (default)
- Redis + Celery distributed workers
- FastAPI Security utilities (OAuth2 bearer + API key header)

## Dataset Ingestion Pipeline

Features:

- Upload dataset metadata and source references
- Streaming dataset loading for large datasets
- Dataset sharding for distributed training
- Distributed training compatibility via shard resolution by `rank/world_size`
- Dataset caching + preprocessing outputs

Supported formats:

- JSONL
- Parquet
- HuggingFace datasets

Dataset endpoints:

- `POST /datasets/upload`
- `GET /datasets`
- `POST /datasets/{dataset_id}/preprocess`
- `GET /datasets/{dataset_id}/shard?rank=0&world_size=8`

## Model Registry Service

Supports:

- HuggingFace model references (`source_type=huggingface`)
- Local checkpoint references (`source_type=local_checkpoint`)
- S3-compatible storage for model manifests (when enabled)
- Local filesystem fallback storage (default)

Model metadata fields tracked:

- `model_name`
- `version`
- `parameter_count`
- `architecture`
- `owner`
- `upload_date`

Model endpoints:

- `POST /models/upload`
- `GET /models`
- `POST /models/{model_name}/versions`


## Training Runtime Engine

Features:

- Distributed training runtime orchestration
- DeepSpeed integration config generation
- FSDP support config generation
- Gradient checkpointing toggle
- Mixed precision (`bf16`/`fp16`/`fp32`)
- Checkpoint saving
- CPU offload for quantized QLoRA training

Training methods:

- Supervised fine tuning
- QLoRA training (4-bit quantization via bitsandbytes + PEFT adapters)
- LoRA adapters
- RLHF (future support placeholder)

Training configuration is modeled as a dataclass-backed runtime config and is configurable via API payload (`runtime`).

### Distributed Training Launcher

Supported launchers:

- `torchrun`
- `deepspeed`
- `ray`

Automatically configures:

- rank
- world size
- master address
- communication backend (`nccl`)


## Memory-Efficient Model Loader (AirLLM-style)

Implemented a layer streaming loader with pipeline semantics:

- split model into layer shards
- persist shards on disk
- dynamically load layers during inference
- execution flow: `disk -> CPU -> GPU -> compute -> unload`
- prefetching to overlap loading and compute

Core implementation: `rivinity_nexus.engine.memory_loader.AirLLMStyleLoader`.

## The Real Secret Behind 10T Models

A 10-trillion parameter dense model is unrealistic for practical deployment.

Memory needed for dense FP16:

- `10T × 2 bytes = 20 TB`

Even with 4-bit quantization:

- `10T × 0.5 bytes = 5 TB`

That is still too large for standard deployment footprints.

### Sparse Mixture of Experts (MoE)

Instead of one monolithic dense model, the runtime follows sparse expert activation:

- Total parameters: `10T`
- Experts: `256`
- Active experts per token: `2`
- Active parameters per token: `~30B–80B` (default runtime profile ~`40B`)

Compute path (simplified transformer block):

`Input -> Attention -> Router -> Selected Experts -> Output`

Only selected experts run on each token.

### Why this fits consumer hardware

Rivinity Nexus combines three strategies together:

1. Sparse experts (MoE)
2. Layer streaming
3. Quantization (4-bit)

Resulting active memory footprint is approximately:

- `~40B active params × 0.5 bytes ≈ 20 GB`

This is designed to fit devices such as:

- RTX 3090
- RTX 4090
- A100

You can inspect the runtime MoE profile via `GET /inference/moe-profile`.



### New MoE System Modules

Implemented modules:

- `engine/moe_router.py`
- `engine/expert_loader.py`
- `engine/kv_cache_manager.py`
- `engine/layer_streamer.py`
- `scheduler/expert_scheduler.py`
- `distributed/expert_rpc.py`

These provide top-k routing, dynamic expert loading, paged/quantized KV cache control, NVMe-to-GPU streaming semantics, expert-to-node assignment, and async remote expert RPC execution.

### Additional Optimization Layers

To push toward 10T-class total parameter systems, Rivinity Nexus adds advanced sparse-system layers:

#### Expert Sharding

Experts are distributed across nodes (example 256 experts / 4 nodes):

- Node1 -> experts 0-63
- Node2 -> experts 64-127
- Node3 -> experts 128-191
- Node4 -> experts 192-255

Router output determines which node executes each expert.

#### Token Routing

Router decisions are token-level, for example:

- token -> expert 17
- token -> expert 103

This requires fast distributed token routing across expert shards.

Supported integration targets:

- DeepSpeed MoE
- Megatron MoE
- Tutel

#### KV Cache Compression

KV cache growth is controlled through:

- quantized KV cache (8-bit / 4-bit)
- paged KV cache patterns (vLLM style)

#### Dynamic Expert Loading (Rivinity unique layer)

Experts can be stored on NVMe and loaded dynamically:

- NVMe -> RAM -> GPU

This enables hosting very large expert pools beyond resident GPU memory.

### Rivinity Nexus Advanced Architecture

```text
                Rivinity Nexus

       ┌────────────────────────────┐
       │        Control Plane        │
       │                             │
       │  FastAPI API                │
       │  Model registry             │
       │  Dataset registry           │
       │  Job scheduler              │
       └───────────────┬─────────────┘
                       │
                       ▼
       ┌────────────────────────────┐
       │         Compute Plane       │
       │                             │
       │  Training Engine            │
       │  DeepSpeed / FSDP           │
       │                             │
       │  Inference Engine           │
       │  Layer streaming            │
       │  MoE router                 │
       │  Expert loader              │
       │  KV cache manager           │
       └───────────────┬─────────────┘
                       │
                       ▼
       ┌────────────────────────────┐
       │        Hardware Layer       │
       │                             │
       │ GPU cluster                 │
       │ CPU memory pool             │
       │ NVMe storage                │
       └────────────────────────────┘
```

## Inference Engine

Features:

- Streaming model execution
- KV cache support
- Batch inference
- Token streaming (SSE)

Supported backends:

- AirLLM runtime
- vLLM runtime

Automatic backend selection:

- `auto` picks vLLM on CUDA + vLLM availability
- falls back to AirLLM-style layer streaming loader otherwise

Inference endpoints:

- `POST /inference/generate`
- `POST /inference/stream`

## GPU Scheduler

Features:

- vendor node registration (`vendor_name`, `node_ip`, `gpu_type`, `vram`, `price_per_hour`)
- detect GPU nodes
- track VRAM usage
- allocate GPU resources
- schedule jobs across nodes

Scheduling strategies:

- `least_loaded`
- `vram_aware`
- `cost_aware`

Returns a node allocation plan via `POST /gpu/schedule`.

Vendor endpoints:

- `POST /gpu/vendors/register`
- `GET /gpu/vendors`

## Distributed Job Queue System

- Broker/result backend: Redis + Celery
- Job types:
  - `training_job`
  - `inference_job`
  - `dataset_processing_job`
- Worker behavior:
  - Poll queue (Celery workers)
  - Execute typed tasks
  - Send progress updates to Redis Pub/Sub channel `rivinity:jobs:progress`
- Job state tracking in DB (`queue_jobs`): `PENDING`, `RUNNING`, `FAILED`, `COMPLETED`


## Experiment Tracking

Integrated MLflow experiment tracking for training runs.

Tracked artifacts and metadata:

- hyperparameters (runtime config)
- training metrics (e.g. `training_loss`)
- model checkpoints (logged from output checkpoint directory)
- datasets (dataset URI tag + local dataset artifact when path exists)

Configuration (environment variables):

- `EXPERIMENT_TRACKING_ENABLED`
- `EXPERIMENT_TRACKING_BACKEND` (`mlflow`)
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`


## Kubernetes Deployment

Kubernetes manifests are provided under `k8s/` for:

- API service + deployment (`api-deployment.yaml`)
- Worker deployment with GPU scheduling (`worker-deployment.yaml`)
- GPU node pool (`gpu-nodepool.yaml`, Karpenter NodePool)
- Redis deployment + service (`redis.yaml`)
- PostgreSQL StatefulSet + service (`postgres.yaml`)

Apply with:

```bash
kubectl apply -k k8s/
```

> Update `k8s/secret.example.yaml` with real credentials/tokens before deploying.

## Monitoring and Prometheus Metrics

Prometheus metrics are exported at `GET /metrics`.

Tracked metrics include:

- GPU utilization (`rivinity_gpu_utilization_percent`)
- VRAM usage (`rivinity_gpu_vram_usage_gb`)
- Tokens per second (`rivinity_tokens_per_second`)
- Training loss (`rivinity_training_loss`)
- Queue latency (`rivinity_queue_latency_seconds`)

## Run

```bash
cp .env.example .env
docker compose up --build
```

Compose services included:

- `api` (FastAPI server)
- `worker` (Celery worker)
- `scheduler` (Celery beat scheduler)
- `database` (PostgreSQL)
- `redis` (Redis broker/result backend)

Open docs at `http://localhost:8000/docs`.
