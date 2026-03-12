from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

import torch

from rivinity_nexus.config.settings import get_settings
from rivinity_nexus.core.resilience import retry_call
from rivinity_nexus.engine.memory_loader import AirLLMStyleLoader, LayerShardStore

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None


class InferenceBackend(str, Enum):
    auto = "auto"
    airllm = "airllm"
    vllm = "vllm"


@dataclass
class GenerationRequest:
    model_name: str
    prompts: list[str]
    max_new_tokens: int = 128
    temperature: float = 0.7
    backend: InferenceBackend = InferenceBackend.auto
    use_kv_cache: bool = True


@dataclass
class GenerationResult:
    backend: InferenceBackend
    outputs: list[str]
    token_counts: list[int] = field(default_factory=list)


class BaseRuntime:
    def generate(self, req: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def stream_tokens(self, req: GenerationRequest) -> Generator[str, None, None]:
        raise NotImplementedError


class AirLLMRuntime(BaseRuntime):
    def __init__(self, model_storage_path: str) -> None:
        self.model_storage_path = model_storage_path
        self.loaders: dict[str, AirLLMStyleLoader] = {}
        self.kv_cache: dict[tuple[str, str], str] = {}

    def _build_loader(self, model_name: str) -> AirLLMStyleLoader:
        shard_dir = str(Path(self.model_storage_path) / "shards" / model_name)
        return AirLLMStyleLoader(
            shard_store=LayerShardStore(root_dir=shard_dir),
            gpu_device="cuda" if torch.cuda.is_available() else "cpu",
            prefetch_depth=2,
            max_gpu_layers=2,
        )

    def _get_loader(self, model_name: str) -> AirLLMStyleLoader:
        if model_name not in self.loaders:
            self.loaders[model_name] = self._build_loader(model_name)
        return self.loaders[model_name]

    def _ensure_demo_shards(self, loader: AirLLMStyleLoader) -> int:
        existing = list(Path(loader.shard_store.root_dir).glob("layer_*.pt"))
        if existing:
            return len(existing)
        layer_state_dicts = [{"weight": torch.ones(1) * idx} for idx in range(8)]
        loader.split_and_store_model_layers(layer_state_dicts)
        return len(layer_state_dicts)

    def _compute_tokens(self, loader: AirLLMStyleLoader, total_layers: int) -> list[str]:
        return loader.run_layerwise_inference(total_layers=total_layers, compute_fn=lambda idx, _: f"T{idx}")

    def generate(self, req: GenerationRequest) -> GenerationResult:
        loader = self._get_loader(req.model_name)
        total_layers = self._ensure_demo_shards(loader)

        outputs: list[str] = []
        token_counts: list[int] = []

        for prompt in req.prompts:
            cache_key = (req.model_name, prompt)
            if req.use_kv_cache and cache_key in self.kv_cache:
                out = self.kv_cache[cache_key]
            else:
                tokens = self._compute_tokens(loader, total_layers)
                out = f"[airllm] {' '.join(tokens)} :: {prompt[:80]}"
                if req.use_kv_cache:
                    self.kv_cache[cache_key] = out
            outputs.append(out)
            token_counts.append(min(req.max_new_tokens, total_layers))

        return GenerationResult(backend=InferenceBackend.airllm, outputs=outputs, token_counts=token_counts)

    def stream_tokens(self, req: GenerationRequest) -> Generator[str, None, None]:
        loader = self._get_loader(req.model_name)
        total_layers = self._ensure_demo_shards(loader)
        tokens = self._compute_tokens(loader, total_layers)
        for token in tokens[: req.max_new_tokens]:
            yield token


class VLLMRuntime(BaseRuntime):
    def __init__(self) -> None:
        self.engines: dict[str, LLM] = {}

    def _get_engine(self, model_name: str):
        if model_name not in self.engines:
            if LLM is None:
                raise RuntimeError("vLLM runtime unavailable")
            self.engines[model_name] = LLM(model=model_name)
        return self.engines[model_name]

    def generate(self, req: GenerationRequest) -> GenerationResult:
        engine = self._get_engine(req.model_name)
        params = SamplingParams(max_tokens=req.max_new_tokens, temperature=req.temperature)
        generations = engine.generate(req.prompts, params)
        outputs = [g.outputs[0].text for g in generations]
        token_counts = [len(g.outputs[0].token_ids) for g in generations]
        return GenerationResult(backend=InferenceBackend.vllm, outputs=outputs, token_counts=token_counts)

    def stream_tokens(self, req: GenerationRequest) -> Generator[str, None, None]:
        # For scaffold purposes expose simple tokenized chunks from first prompt.
        result = self.generate(GenerationRequest(**{**req.__dict__, "prompts": [req.prompts[0]]}))
        for tok in result.outputs[0].split()[: req.max_new_tokens]:
            yield tok


class InferenceEngine:
    def __init__(self) -> None:
        settings = get_settings()
        self.airllm = AirLLMRuntime(settings.model_storage_path)
        self.vllm = VLLMRuntime()
        self.logger = logging.getLogger("rivinity_nexus.inference")

    def _select_backend(self, preferred: InferenceBackend) -> InferenceBackend:
        if preferred != InferenceBackend.auto:
            return preferred
        if torch.cuda.is_available() and LLM is not None:
            return InferenceBackend.vllm
        return InferenceBackend.airllm

    def generate(self, req: GenerationRequest) -> GenerationResult:
        backend = self._select_backend(req.backend)
        if backend == InferenceBackend.vllm:
            try:
                return retry_call(lambda: self.vllm.generate(req), attempts=2, delay_seconds=0.1, op_name="vllm_generate")
            except Exception as exc:
                self.logger.warning("vllm_fallback_to_airllm", extra={"model": req.model_name, "error": str(exc)})
        return retry_call(lambda: self.airllm.generate(req), attempts=2, delay_seconds=0.1, op_name="airllm_generate")

    def stream_tokens(self, req: GenerationRequest) -> Generator[str, None, None]:
        backend = self._select_backend(req.backend)
        if backend == InferenceBackend.vllm:
            try:
                yield from self.vllm.stream_tokens(req)
                return
            except Exception as exc:
                self.logger.warning("vllm_stream_fallback_to_airllm", extra={"model": req.model_name, "error": str(exc)})
        yield from self.airllm.stream_tokens(req)
