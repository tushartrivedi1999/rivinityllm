from collections.abc import Generator
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

import torch

from rivinity_nexus.config.settings import get_settings
from rivinity_nexus.core.resilience import retry_call
from rivinity_nexus.distributed.expert_rpc import ExpertRPCClient
from rivinity_nexus.engine.expert_loader import ExpertLoader
from rivinity_nexus.engine.kv_cache_manager import KVCacheManager
from rivinity_nexus.engine.layer_streamer import LayerStreamer
from rivinity_nexus.engine.memory_loader import AirLLMStyleLoader, LayerShardStore
from rivinity_nexus.engine.moe import ExpertShardPlanner, MoEArchitecture
from rivinity_nexus.engine.moe_router import MoERouter
from rivinity_nexus.scheduler.expert_scheduler import ExpertScheduler

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
        self.router = MoERouter(num_experts=256, top_k=2)
        self.sharder = ExpertShardPlanner(num_experts=256, nodes=4)
        self.expert_loader = ExpertLoader(
            str(Path(model_storage_path) / "experts"),
            cpu_cache_size=128,
            max_gpu_experts=8,
            prefetch_workers=4,
        )
        self.kv_manager = KVCacheManager(max_gpu_entries=512, quantization_bits=4, paged=True)
        self.layer_streamer = LayerStreamer()
        self.expert_scheduler = ExpertScheduler()
        self.rpc = ExpertRPCClient()

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


    def _execute_rpc(self, assignment: dict[int, str], tokens: list[str]) -> dict[int, list[str]]:
        try:
            asyncio.get_running_loop()
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self.rpc.execute_distributed(assignment, tokens))
            finally:
                new_loop.close()
        except RuntimeError:
            return asyncio.run(self.rpc.execute_distributed(assignment, tokens))

    def generate(self, req: GenerationRequest) -> GenerationResult:
        loader = self._get_loader(req.model_name)
        total_layers = self._ensure_demo_shards(loader)

        outputs: list[str] = []
        token_counts: list[int] = []

        for prompt in req.prompts:
            cache_key = (req.model_name, prompt)
            cache_key_str = f"{cache_key[0]}::{cache_key[1]}"
            cached = self.kv_manager.get(cache_key_str) if req.use_kv_cache else None
            if cached:
                out = cached
            else:
                tokens = self._compute_tokens(loader, total_layers)
                routing = self.router.route([float(len(prompt) or 1)])
                expert_ids = routing.expert_ids
                self.expert_loader.stream_for_compute(expert_ids, lookahead_experts=[(eid + 1) % 256 for eid in expert_ids])
                node_map = {node: experts for node, experts in self.sharder.shard_map().items()}
                available_nodes = [n for n in node_map.keys()]
                assignment = self.expert_scheduler.assign(expert_ids, available_nodes)
                rpc_outputs = self._execute_rpc(assignment, tokens)
                expert_paths = [self.expert_loader.expert_path(e) for e in expert_ids]
                kv_profile = self.kv_manager.describe()
                out = (
                    f"[airllm-moe experts={expert_ids} nodes={assignment} kv={kv_profile['quantization_bits']}bit stream={self.layer_streamer.describe()}] "
                    f"{' '.join(tokens)} :: {prompt[:80]} :: experts={expert_paths} :: rpc={rpc_outputs}"
                )
                for expert_id in expert_ids:
                    self.expert_loader.evict_from_gpu(expert_id)
                if req.use_kv_cache:
                    self.kv_manager.put(cache_key_str, out)
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
        self.moe_arch = MoEArchitecture()

    def _select_backend(self, preferred: InferenceBackend) -> InferenceBackend:
        if preferred != InferenceBackend.auto:
            return preferred
        if torch.cuda.is_available() and LLM is not None:
            return InferenceBackend.vllm
        return InferenceBackend.airllm


    def get_moe_runtime_profile(self) -> dict:
        sharder = ExpertShardPlanner(num_experts=self.moe_arch.num_experts, nodes=4)
        kv_profile = KVCacheManager(max_gpu_entries=512, quantization_bits=self.moe_arch.quantization_bits, paged=True).describe()
        return {
            "total_params_trillion": self.moe_arch.total_params_trillion,
            "num_experts": self.moe_arch.num_experts,
            "top_k_experts": self.moe_arch.top_k_experts,
            "dense_fp16_memory_tb": self.moe_arch.dense_fp16_memory_tb(),
            "dense_quantized_memory_tb": self.moe_arch.dense_quantized_memory_tb(),
            "active_params_billion": self.moe_arch.active_params_billion,
            "active_memory_gb": self.moe_arch.active_memory_gb(),
            "estimated_compute_reduction_x": self.moe_arch.estimate_compute_reduction(),
            "expert_shards": sharder.shard_map(),
            "kv_cache": kv_profile,
            "dynamic_expert_loading": "NVMe->CPU RAM->GPU->Compute->Evict",
        }

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
