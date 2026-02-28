# Getting Started

## 1) Environment
- Python 3.11+
- Virtual environment recommended

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## 2) Run API
```bash
uvicorn rivinityllm_tune.api.main:app --reload
```

## 3) Verify health
```bash
curl http://127.0.0.1:8000/health
```
Expected response:
```json
{"status":"ok"}
```

## 4) Basic workflow
1. Create user via `/auth/signup`
2. Login via `/auth/login`
3. Submit RL job via `/v1/train/rl`
4. Estimate deployment via `/v1/inference/plan`
5. Call generation endpoint via `/v1/inference/generate`

## 5) Planner-first workflow for large models
For large deployments (200B+), always run planner first and validate:
- `gpu_count`
- `pipeline_parallelism`
- `min_nvme_read_gbps`

Then use those results to set cluster topology and storage profile.
