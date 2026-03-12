import pytest

pytest.importorskip("torch")

from rivinity_nexus.engine.inference import GenerationRequest, InferenceBackend, InferenceEngine, GenerationResult


class _BrokenVLLM:
    def generate(self, req):
        _ = req
        raise RuntimeError("vllm down")


class _Air:
    def generate(self, req):
        return GenerationResult(backend=InferenceBackend.airllm, outputs=["fallback"], token_counts=[1])


def test_inference_engine_falls_back_to_airllm() -> None:
    engine = InferenceEngine()
    engine.vllm = _BrokenVLLM()
    engine.airllm = _Air()

    req = GenerationRequest(model_name="demo", prompts=["hello"], backend=InferenceBackend.vllm)
    result = engine.generate(req)

    assert result.backend == InferenceBackend.airllm
    assert result.outputs == ["fallback"]


def test_moe_profile_contains_advanced_layers() -> None:
    engine = InferenceEngine()
    profile = engine.get_moe_runtime_profile()
    assert profile["dense_fp16_memory_tb"] == 20.0
    assert profile["dense_quantized_memory_tb"] == 5.0
    assert profile["active_memory_gb"] == 20.0
    assert "node-1" in profile["expert_shards"]
    assert profile["kv_cache"]["quantization_bits"] == 4
    assert profile["dynamic_expert_loading"] == "NVMe->CPU RAM->GPU->Compute->Evict"
