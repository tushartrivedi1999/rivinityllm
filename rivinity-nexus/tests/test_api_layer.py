import os
from pathlib import Path

from fastapi.testclient import TestClient

TEST_DB = Path("test_nexus.db")
TEST_DATASET = Path("test_dataset.jsonl")
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.absolute()}"
os.environ["JWT_SECRET_KEY"] = "test-secret"
os.environ["API_V1_PREFIX"] = ""

from rivinity_nexus.main import app  # noqa: E402
from rivinity_nexus.core.database import SessionLocal  # noqa: E402
from rivinity_nexus.models.entities import User, UserRole  # noqa: E402


class _Task:
    id = "task-test-123"


class _TaskProxy:
    @staticmethod
    def delay(**kwargs):
        _ = kwargs
        return _Task()


class _QueueTaskProxy:
    @staticmethod
    def delay(**kwargs):
        _ = kwargs
        return _Task()


def _signup_and_login(client: TestClient, email: str) -> tuple[dict[str, str], str]:
    password = "strongpass123"
    reg = client.post(
        "/auth/signup",
        json={"email": email, "full_name": "Owner", "password": password},
    )
    assert reg.status_code in (201, 409)

    login = client.post("/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    body = login.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["refresh_token"]


def test_auth_tokens_api_keys_and_rbac(monkeypatch) -> None:
    if TEST_DB.exists():
        TEST_DB.unlink()
    if TEST_DATASET.exists():
        TEST_DATASET.unlink()
    TEST_DATASET.write_text("{\"text\": \"hello\"}\n{\"text\": \"world\"}\n")

    from rivinity_nexus.engine import job_queue as job_queue_engine
    from rivinity_nexus.engine import training as training_engine

    monkeypatch.setattr(training_engine, "train_model", _TaskProxy())
    monkeypatch.setattr(job_queue_engine, "training_job", _QueueTaskProxy())
    monkeypatch.setattr(job_queue_engine, "inference_job", _QueueTaskProxy())
    monkeypatch.setattr(job_queue_engine, "dataset_processing_job", _QueueTaskProxy())

    with TestClient(app) as client:
        bearer_headers, refresh_token = _signup_and_login(client, "owner@example.com")

        refreshed = client.post("/auth/refresh", json={"refresh_token": refresh_token})
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"]

        created_key = client.post("/auth/api-keys", json={"name": "automation"}, headers=bearer_headers)
        assert created_key.status_code == 201
        raw_api_key = created_key.json()["key"]

        listed_keys = client.get("/auth/api-keys", headers=bearer_headers)
        assert listed_keys.status_code == 200
        assert len(listed_keys.json()) == 1

        upload = client.post(
            "/models/upload",
            json={"model_name": "mistral-lite", "version": "v1", "parameter_count": 7000000000, "architecture": "MistralForCausalLM", "source_type": "huggingface", "source_uri": "hf://mistralai/Mistral-7B-v0.1"},
            headers={"X-API-Key": raw_api_key},
        )
        assert upload.status_code == 201
        model_id = upload.json()["id"]

        models_list = client.get("/models", headers={"X-API-Key": raw_api_key})
        assert models_list.status_code == 200
        assert len(models_list.json()) == 1

        new_version = client.post(
            "/models/mistral-lite/versions",
            json={"parameter_count": 7000000000, "architecture": "MistralForCausalLM", "source_type": "local_checkpoint", "source_uri": "/checkpoints/mistral-lite-v2"},
            headers={"X-API-Key": raw_api_key},
        )
        assert new_version.status_code == 201
        assert new_version.json()["version"] == "v2"


        dataset_upload = client.post(
            "/datasets/upload",
            json={"dataset_name": "toy-ds", "version": "v1", "format": "jsonl", "source_uri": str(TEST_DATASET)},
            headers={"X-API-Key": raw_api_key},
        )
        assert dataset_upload.status_code == 201
        dataset_id = dataset_upload.json()["id"]

        dataset_pre = client.post(
            f"/datasets/{dataset_id}/preprocess",
            json={"shard_count": 2},
            headers={"X-API-Key": raw_api_key},
        )
        assert dataset_pre.status_code == 200
        assert dataset_pre.json()["status"] == "preprocessed"

        dataset_shard = client.get(
            f"/datasets/{dataset_id}/shard",
            params={"rank": 1, "world_size": 2},
            headers={"X-API-Key": raw_api_key},
        )
        assert dataset_shard.status_code == 200

        training = client.post(
            "/training/start",
            json={"model_id": model_id, "dataset_uri": "s3://bucket/data", "max_steps": 10, "runtime": {"method": "qlora", "launcher": "deepspeed", "deepspeed_enabled": True, "gradient_checkpointing": True, "qlora_4bit": True, "cpu_offload": True, "lora_r": 16}},
            headers={"X-API-Key": raw_api_key},
        )
        assert training.status_code == 202

        inference = client.post(
            "/inference/generate",
            json={"model_name": "mistral-lite", "prompts": ["Hello", "Hi"], "max_new_tokens": 5, "backend": "auto", "use_kv_cache": True},
            headers={"X-API-Key": raw_api_key},
        )
        assert inference.status_code == 200
        assert len(inference.json()["outputs"]) == 2

        register_vendor_node = client.post(
            "/gpu/vendors/register",
            json={
                "vendor_name": "rivinity-cloud",
                "node_ip": "10.0.0.5",
                "gpu_type": "A100",
                "vram": 80,
                "price_per_hour": 3.25,
                "total_gpus": 4,
                "available_gpus": 3,
            },
            headers={"X-API-Key": raw_api_key},
        )
        assert register_vendor_node.status_code == 201

        vendor_nodes = client.get("/gpu/vendors", headers={"X-API-Key": raw_api_key})
        assert vendor_nodes.status_code == 200
        assert len(vendor_nodes.json()) == 1

        available = client.get("/gpu/available", headers={"X-API-Key": raw_api_key})
        assert available.status_code == 200
        assert available.json()[0]["vendor"] == "rivinity-cloud"

        schedule = client.post(
            "/gpu/schedule",
            json={"required_gpus": 2, "required_vram_gb": 32, "strategy": "least_loaded", "gpu_type": "A100"},
            headers={"X-API-Key": raw_api_key},
        )
        assert schedule.status_code == 200
        assert schedule.json()["total_allocated_gpus"] == 2
        assert schedule.json()["allocations"][0]["vendor_name"] == "rivinity-cloud"

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        body = metrics.text
        assert "rivinity_gpu_utilization_percent" in body
        assert "rivinity_gpu_vram_usage_gb" in body
        assert "rivinity_tokens_per_second" in body

        queued = client.post(
            "/jobs/submit",
            json={"job_type": "dataset_processing_job", "payload": {"dataset_uri": "s3://bucket/raw"}},
            headers={"X-API-Key": raw_api_key},
        )
        assert queued.status_code == 202

        queue_get = client.get(f"/jobs/{queued.json()['id']}", headers={"X-API-Key": raw_api_key})
        assert queue_get.status_code == 200
        assert queue_get.json()["state"] == "PENDING"

        forbidden = client.get("/auth/users", headers={"X-API-Key": raw_api_key})
        assert forbidden.status_code == 403

        db = SessionLocal()
        user = db.query(User).filter(User.email == "owner@example.com").first()
        user.role = UserRole.admin
        db.commit()
        db.close()

        admin_login = client.post("/auth/login", json={"email": "owner@example.com", "password": "strongpass123"})
        admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}
        admin_users = client.get("/auth/users", headers=admin_headers)
        assert admin_users.status_code == 200
        assert len(admin_users.json()) >= 1

        logout = client.post("/auth/logout", json={"refresh_token": refresh_token})
        assert logout.status_code == 204

        refresh_after_logout = client.post("/auth/refresh", json={"refresh_token": refresh_token})
        assert refresh_after_logout.status_code == 401

    if TEST_DB.exists():
        TEST_DB.unlink()
    if TEST_DATASET.exists():
        TEST_DATASET.unlink()
