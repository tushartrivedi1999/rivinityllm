import time

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

metrics_router = APIRouter()

request_counter = Counter("rivinity_requests_total", "Total API requests")

# GPU metrics
gpu_utilization_percent = Gauge(
    "rivinity_gpu_utilization_percent",
    "Current GPU utilization percentage by node",
    ["vendor", "node_id", "gpu_type"],
)
vram_usage_gb = Gauge(
    "rivinity_gpu_vram_usage_gb",
    "Current VRAM usage in GB by node",
    ["vendor", "node_id", "gpu_type"],
)

# Inference metrics
tokens_per_second = Histogram(
    "rivinity_tokens_per_second",
    "Inference throughput in generated tokens per second",
    ["backend", "model_name"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")),
)

# Training metrics
training_loss = Gauge(
    "rivinity_training_loss",
    "Most recent training loss value",
    ["method", "model_name"],
)

# Queue metrics
queue_latency_seconds = Histogram(
    "rivinity_queue_latency_seconds",
    "Latency between queue submission and start of execution",
    ["job_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, float("inf")),
)


def record_gpu_node_metrics(vendor: str, node_id: str, gpu_type: str, total_gpus: int, free_gpus: int, total_vram_gb: int, used_vram_gb: int) -> None:
    utilization = 0.0
    if total_gpus > 0:
        utilization = ((total_gpus - free_gpus) / total_gpus) * 100.0
    gpu_utilization_percent.labels(vendor=vendor, node_id=node_id, gpu_type=gpu_type).set(round(utilization, 4))
    vram_usage_gb.labels(vendor=vendor, node_id=node_id, gpu_type=gpu_type).set(float(used_vram_gb if used_vram_gb >= 0 else 0))


def record_tokens_per_second(backend: str, model_name: str, token_count: int, elapsed_seconds: float) -> None:
    safe_elapsed = max(elapsed_seconds, 1e-6)
    tps = token_count / safe_elapsed
    tokens_per_second.labels(backend=backend, model_name=model_name).observe(tps)


def record_training_loss(method: str, model_name: str, loss: float) -> None:
    training_loss.labels(method=method, model_name=model_name).set(loss)


def record_queue_latency(job_type: str, latency_seconds: float) -> None:
    queue_latency_seconds.labels(job_type=job_type).observe(max(latency_seconds, 0.0))


def timed() -> float:
    return time.perf_counter()


@metrics_router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    request_counter.inc()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
