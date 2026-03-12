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
