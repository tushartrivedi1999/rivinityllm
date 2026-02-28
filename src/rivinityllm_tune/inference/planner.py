"""Planning utilities for ultra-large model inference.

The planner is inspired by low-VRAM serving techniques used by systems like AirLLM:
- layer-wise weight streaming/offloading
- aggressive CPU/NVMe backing store
- runtime prefetch windows to keep compute fed

This module extends those ideas to the 1000B class by generating a deployment plan
that includes sharding, host memory, and NVMe throughput constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil

BYTES_PER_GIB = 1024**3
SUPPORTED_QUANT_BITS = frozenset({2, 3, 4, 8, 16})
GPU_ACTIVE_RATIO = 0.10
CPU_CACHE_RATIO = 0.20
NVME_OFFLOAD_RATIO = 0.70
GPU_SAFETY_UTILIZATION = 0.80


@dataclass(frozen=True)
class InferencePlan:
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


class PlanError(ValueError):
    """Raised when a model plan cannot be generated."""


def estimate_weight_footprint_gb(parameter_count_b: float, quantization_bits: int) -> float:
    if parameter_count_b <= 0:
        raise PlanError("parameter_count_b must be > 0")
    if quantization_bits not in SUPPORTED_QUANT_BITS:
        raise PlanError("quantization_bits must be one of 2, 3, 4, 8, 16")
    bytes_per_param = quantization_bits / 8
    return parameter_count_b * 1_000_000_000 * bytes_per_param / BYTES_PER_GIB


@lru_cache(maxsize=1024)
def build_inference_plan(
    parameter_count_b: float,
    quantization_bits: int,
    target_gpu_vram_gb: float,
    sequence_length: int,
    batch_size: int,
) -> InferencePlan:
    """Generate a practical serving plan for very large models.

    Heuristic assumptions:
    - 70% weights on NVMe cold storage, 20% in CPU cache, 10% active shard slices.
    - tensor parallelism is primary scaling vector; pipeline added for 1000B-class.
    - prefetch window grows with sequence length to hide storage latency.
    """

    if target_gpu_vram_gb <= 0:
        raise PlanError("target_gpu_vram_gb must be > 0")
    if sequence_length < 128:
        raise PlanError("sequence_length must be >= 128")
    if batch_size < 1:
        raise PlanError("batch_size must be >= 1")

    weight_size_gb = estimate_weight_footprint_gb(parameter_count_b, quantization_bits)

    active_weight_slice_gb = weight_size_gb * GPU_ACTIVE_RATIO
    gpu_capacity_effective = target_gpu_vram_gb * GPU_SAFETY_UTILIZATION
    gpu_count = max(1, ceil(active_weight_slice_gb / gpu_capacity_effective))

    tensor_parallelism = max(1, min(64, gpu_count))
    pipeline_parallelism = 1 if parameter_count_b < 300 else max(2, ceil(parameter_count_b / 250))

    cpu_cache_gb = weight_size_gb * CPU_CACHE_RATIO
    nvme_offload_gb = weight_size_gb * NVME_OFFLOAD_RATIO

    traffic_multiplier = (sequence_length / 2048) * (1 + (batch_size - 1) * 0.15)
    prefetch_window_layers = max(2, min(16, ceil(2 * traffic_multiplier)))

    min_nvme_read_gbps = round(max(2.0, (nvme_offload_gb / 400) * traffic_multiplier), 2)

    if parameter_count_b >= 900:
        strategy = "hierarchical_offload+tp+pp+speculative_prefetch"
    elif parameter_count_b >= 200:
        strategy = "hierarchical_offload+tp+pp"
    else:
        strategy = "cpu_offload+paged_weights"

    return InferencePlan(
        parameter_count_b=parameter_count_b,
        quantization_bits=quantization_bits,
        weight_size_gb=round(weight_size_gb, 2),
        target_gpu_vram_gb=target_gpu_vram_gb,
        gpu_count=gpu_count,
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        cpu_cache_gb=round(cpu_cache_gb, 2),
        nvme_offload_gb=round(nvme_offload_gb, 2),
        prefetch_window_layers=prefetch_window_layers,
        min_nvme_read_gbps=min_nvme_read_gbps,
        strategy=strategy,
    )
