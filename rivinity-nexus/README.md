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
