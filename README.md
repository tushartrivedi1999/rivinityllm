# RivinityLLM Tune

RivinityLLM Tune is a unified platform blueprint for:
- **Low-VRAM large model inference** with smart offload strategies.
- **RL-based post-training pipelines** for custom agent behavior.
- **1000B-class deployment planning** via hierarchical offload + parallelism recommendations.

## Current capabilities
- FastAPI backend scaffold with:
  - Health endpoint
  - Signup/login authentication endpoints
  - RL job submission endpoint
  - Inference generation endpoint
  - Inference deployment planner endpoint for ultra-large models (`/v1/inference/plan`)
- Python SDK for client integration
- Tests for API behavior and planner logic
- Multi-document technical documentation

## Repository structure
- `src/rivinityllm_tune/api`: API application entrypoint and routes
- `src/rivinityllm_tune/auth`: auth schemas, security, and user store
- `src/rivinityllm_tune/training`: RL training request/response contracts
- `src/rivinityllm_tune/inference`: inference contracts + 1000B planning logic
- `sdk/rivinityllm_sdk`: Python SDK
- `docs/architecture.md`: system architecture
- `docs/getting-started.md`: setup and first run
- `docs/api-reference.md`: endpoints and payloads
- `docs/operational-guidelines.md`: production Do's/Don'ts

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn rivinityllm_tune.api.main:app --reload
```

Open http://127.0.0.1:8000/docs for API docs.

## SDK examples
```python
from rivinityllm_sdk import RivinityClient

with RivinityClient("http://127.0.0.1:8000") as client:
    client.signup("you@example.com", "StrongPass123")
    client.login("you@example.com", "StrongPass123")

    job = client.submit_rl_training(
        base_model="meta-llama/Meta-Llama-3.1-8B",
        environment="tool-use-v1",
        algorithm="ppo",
        max_steps=5000,
    )

    plan = client.plan_inference(
        parameter_count_b=1000,
        quantization_bits=4,
        target_gpu_vram_gb=8,
        sequence_length=8192,
        batch_size=2,
    )

    generation = client.generate(
        model_id="meta-llama/Meta-Llama-3.1-8B",
        prompt="Write a sustainable AI infrastructure checklist.",
        memory_budget_gb=8,
    )

print(job, plan["strategy"], generation["strategy"])
```

## Do's and Don'ts

### ✅ Do
- Use `/v1/inference/plan` before provisioning hardware for new model sizes.
- Keep quantization explicit (2/3/4/8/16 bits) and validate quality offline.
- Start with small batch size and scale up while monitoring NVMe throughput.
- Treat this repository as a **control-plane + planning foundation** until full runtime is added.
- Use deterministic, reproducible payloads for planner and training API calls.
- Keep secrets out of code and set secure JWT values in production.

### ❌ Don't
- Don’t assume this planner alone is enough for production runtime guarantees.
- Don’t use the in-memory user store in production.
- Don’t expose `/auth/*` publicly without rate limiting and brute-force protections.
- Don’t run 1000B deployments without validating IO path (NVMe bandwidth and tail latency).
- Don’t trust synthetic/demo output from `/v1/inference/generate` as model quality proof.
- Don’t skip observability (metrics, traces, structured logs, audit trails).

## Detailed documentation
- [Architecture](docs/architecture.md)
- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Operational Guidelines](docs/operational-guidelines.md)

## Upstream analysis note
Attempted direct live access to:
- `https://github.com/rllm-org/rllm.git`
- `https://github.com/lyogavin/airllm.git`

This environment blocks outbound GitHub access (`CONNECT tunnel failed, response 403`).
The implementation therefore codifies known patterns and extends them with explicit 1000B planning primitives.
