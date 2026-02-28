from fastapi.testclient import TestClient

from rivinityllm_tune.api.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signup_login_and_inference() -> None:
    client = TestClient(app)

    signup = client.post(
        "/auth/signup", json={"email": "founder@rivinity.ai", "password": "SecurePass123"}
    )
    assert signup.status_code == 200

    login = client.post(
        "/auth/login", json={"email": "founder@rivinity.ai", "password": "SecurePass123"}
    )
    assert login.status_code == 200
    assert "access_token" in login.json()

    inference = client.post(
        "/v1/inference/generate",
        json={
            "model_id": "llama-3.1-70b",
            "prompt": "hello",
            "max_new_tokens": 16,
            "memory_budget_gb": 8,
        },
    )
    assert inference.status_code == 200
    payload = inference.json()
    assert payload["strategy"] == "cpu_offload+paged_weights"
    assert "hello" in payload["output"]


def test_rl_job_id_is_stable() -> None:
    client = TestClient(app)
    payload = {
        "base_model": "meta-llama/Meta-Llama-3.1-8B",
        "environment": "tool-use-v1",
        "algorithm": "ppo",
        "max_steps": 5000,
    }

    first = client.post("/v1/train/rl", json=payload)
    second = client.post("/v1/train/rl", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]


def test_inference_plan_for_1000b() -> None:
    client = TestClient(app)
    plan_response = client.post(
        "/v1/inference/plan",
        json={
            "parameter_count_b": 1000,
            "quantization_bits": 4,
            "target_gpu_vram_gb": 8,
            "sequence_length": 8192,
            "batch_size": 2,
        },
    )
    assert plan_response.status_code == 200
    plan = plan_response.json()
    assert plan["strategy"] == "hierarchical_offload+tp+pp+speculative_prefetch"
    assert plan["weight_size_gb"] > 400
