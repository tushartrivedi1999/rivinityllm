# API Reference

## `GET /health`
Returns API liveliness.

### Response
```json
{"status": "ok"}
```

---

## `POST /auth/signup`
Create local user in in-memory store.

### Request
```json
{
  "email": "founder@rivinity.ai",
  "password": "SecurePass123"
}
```

### Response
```json
{
  "email": "founder@rivinity.ai"
}
```

---

## `POST /auth/login`
Authenticate user and return bearer token.

### Request
```json
{
  "email": "founder@rivinity.ai",
  "password": "SecurePass123"
}
```

### Response
```json
{
  "access_token": "<jwt>",
  "token_type": "bearer"
}
```

---

## `POST /v1/train/rl`
Queue RL post-training job (control-plane placeholder).

### Request
```json
{
  "base_model": "meta-llama/Meta-Llama-3.1-8B",
  "environment": "tool-use-v1",
  "algorithm": "ppo",
  "max_steps": 5000
}
```

### Response
```json
{
  "job_id": "rl-<stable-hash>",
  "status": "queued"
}
```

---

## `POST /v1/inference/generate`
Demo generation endpoint with strategy selection.

### Request
```json
{
  "model_id": "meta-llama/Meta-Llama-3.1-8B",
  "prompt": "hello",
  "max_new_tokens": 32,
  "memory_budget_gb": 8
}
```

### Response
```json
{
  "output": "[demo-output for ...] hello",
  "strategy": "cpu_offload+paged_weights"
}
```

---

## `POST /v1/inference/plan`
Generate deployment plan for ultra-large model serving.

### Request
```json
{
  "parameter_count_b": 1000,
  "quantization_bits": 4,
  "target_gpu_vram_gb": 8,
  "sequence_length": 8192,
  "batch_size": 2
}
```

### Response fields
- `weight_size_gb`: total model weight size estimate
- `gpu_count`: minimum active shard GPU estimate
- `tensor_parallelism`: suggested TP width
- `pipeline_parallelism`: suggested PP depth
- `cpu_cache_gb`: warm CPU cache target
- `nvme_offload_gb`: cold storage target
- `prefetch_window_layers`: planned layer prefetch window
- `min_nvme_read_gbps`: lower-bound sequential throughput target
- `strategy`: selected execution strategy
