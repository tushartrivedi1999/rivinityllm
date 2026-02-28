# RivinityLLM Tune Architecture (v0.3 Foundation)

## What AirLLM is doing (and how)
AirLLM-style systems focus on **inference memory compression by execution design**, not only model compression:
1. Keep full model weights mostly off-GPU (CPU/NVMe).
2. Load only active layer/tensor slices into VRAM when needed.
3. Use prefetch + paging to overlap IO with compute.
4. Favor low-VRAM cards and horizontal orchestration over single massive GPUs.

## RivinityLLM adaptation target
RivinityLLM extends this approach from 70B-class to **1000B-class planning** by adding:
- hierarchical offload planning (GPU + CPU cache + NVMe cold storage)
- tensor + pipeline parallelism recommendations
- throughput planning for storage bottlenecks
- prefetch window sizing based on context length and batch size

## Optimizations in current implementation
- **Deterministic RL job IDs** via SHA-256 seed hashing for idempotent orchestration.
- **Planner caching** via LRU for repeated planning workloads.
- **Centralized constants** for ratio tuning and simpler future calibration.
- **SDK request abstraction** to reduce duplicate networking code and improve maintainability.
- **App factory pattern** to support test isolation and future multi-app bootstrapping.

## Current APIs
- Auth: `/auth/signup`, `/auth/login`
- RL enqueue: `/v1/train/rl`
- Demo text inference: `/v1/inference/generate`
- 1000B deployment planner: `/v1/inference/plan`

## Execution planes
- **Control Plane**: identity, project/job orchestration, policies.
- **Training Plane**: RL PPO/GRPO/DPO workers and environment runtime.
- **Inference Plane**: sharded weight loader + paged prefetch scheduler + offload hierarchy.
- **Observability Plane**: metrics/traces/logs + cost and sustainability analytics.

## Important environment note
Direct live pull of upstream reference code was attempted from this environment but blocked by outbound network policy (403 tunnel failure). The implemented planner follows public architectural patterns and is structured so direct parity work can be added when network access is available.
