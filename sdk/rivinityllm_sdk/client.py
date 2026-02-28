"""Python SDK client for RivinityLLM Tune."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx


class RivinityClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._token: str | None = None

    def __enter__(self) -> "RivinityClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, json: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self._client.request(method, path, json=dict(json))
        response.raise_for_status()
        return response.json()

    def signup(self, email: str, password: str) -> Mapping[str, Any]:
        return self._request("POST", "/auth/signup", {"email": email, "password": password})

    def login(self, email: str, password: str) -> str:
        payload = self._request("POST", "/auth/login", {"email": email, "password": password})
        token = str(payload["access_token"])
        self._token = token
        self._client.headers.update({"Authorization": f"Bearer {token}"})
        return token

    def submit_rl_training(
        self, base_model: str, environment: str, algorithm: str = "ppo", max_steps: int = 1000
    ) -> Mapping[str, Any]:
        return self._request(
            "POST",
            "/v1/train/rl",
            {
                "base_model": base_model,
                "environment": environment,
                "algorithm": algorithm,
                "max_steps": max_steps,
            },
        )

    def generate(
        self, model_id: str, prompt: str, max_new_tokens: int = 128, memory_budget_gb: float = 8.0
    ) -> Mapping[str, Any]:
        return self._request(
            "POST",
            "/v1/inference/generate",
            {
                "model_id": model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "memory_budget_gb": memory_budget_gb,
            },
        )

    def plan_inference(
        self,
        parameter_count_b: float,
        quantization_bits: int = 4,
        target_gpu_vram_gb: float = 8.0,
        sequence_length: int = 4096,
        batch_size: int = 1,
    ) -> Mapping[str, Any]:
        return self._request(
            "POST",
            "/v1/inference/plan",
            {
                "parameter_count_b": parameter_count_b,
                "quantization_bits": quantization_bits,
                "target_gpu_vram_gb": target_gpu_vram_gb,
                "sequence_length": sequence_length,
                "batch_size": batch_size,
            },
        )

    def close(self) -> None:
        self._client.close()
