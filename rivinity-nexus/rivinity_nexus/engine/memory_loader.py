from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LayerShardInfo:
    layer_idx: int
    shard_path: Path


class LayerShardStore:
    """Persist and fetch per-layer shards from disk."""

    def __init__(self, root_dir: str, shard_prefix: str = "layer") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.shard_prefix = shard_prefix

    def _path(self, layer_idx: int) -> Path:
        return self.root_dir / f"{self.shard_prefix}_{layer_idx:05d}.pt"

    def save_layer(self, layer_idx: int, state_dict: dict) -> LayerShardInfo:
        shard_path = self._path(layer_idx)
        torch.save(state_dict, shard_path)
        return LayerShardInfo(layer_idx=layer_idx, shard_path=shard_path)

    def load_layer_to_cpu(self, layer_idx: int) -> dict:
        shard_path = self._path(layer_idx)
        if not shard_path.exists():
            raise FileNotFoundError(f"Layer shard not found: {shard_path}")
        return torch.load(shard_path, map_location="cpu")


class AirLLMStyleLoader:
    """Memory-efficient loader: disk -> CPU -> GPU -> compute -> unload with prefetch."""

    def __init__(
        self,
        shard_store: LayerShardStore,
        gpu_device: str = "cuda",
        prefetch_depth: int = 2,
        max_gpu_layers: int = 2,
    ) -> None:
        self.shard_store = shard_store
        self.gpu_device = gpu_device
        self.prefetch_depth = max(1, prefetch_depth)
        self.max_gpu_layers = max(1, max_gpu_layers)

        self._executor = ThreadPoolExecutor(max_workers=self.prefetch_depth)
        self._prefetch_futures: dict[int, Future] = {}
        self._gpu_cache: OrderedDict[int, dict] = OrderedDict()

    def split_and_store_model_layers(self, layer_state_dicts: list[dict]) -> list[LayerShardInfo]:
        infos: list[LayerShardInfo] = []
        for idx, layer_state in enumerate(layer_state_dicts):
            infos.append(self.shard_store.save_layer(idx, layer_state))
        return infos

    def _to_gpu(self, cpu_state: dict) -> dict:
        if torch.cuda.is_available() and self.gpu_device.startswith("cuda"):
            return {k: v.to(self.gpu_device, non_blocking=True) if torch.is_tensor(v) else v for k, v in cpu_state.items()}
        return cpu_state

    def _unload_from_gpu(self, layer_idx: int) -> None:
        if layer_idx in self._gpu_cache:
            self._gpu_cache.pop(layer_idx, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_gpu_budget(self) -> None:
        while len(self._gpu_cache) > self.max_gpu_layers:
            oldest_idx, _ = self._gpu_cache.popitem(last=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _ = oldest_idx

    def _prefetch(self, layer_idx: int) -> None:
        if layer_idx in self._prefetch_futures or layer_idx in self._gpu_cache:
            return
        self._prefetch_futures[layer_idx] = self._executor.submit(self.shard_store.load_layer_to_cpu, layer_idx)

    def _get_cpu_layer(self, layer_idx: int) -> dict:
        future = self._prefetch_futures.pop(layer_idx, None)
        if future is None:
            return self.shard_store.load_layer_to_cpu(layer_idx)
        return future.result()

    def load_layer(self, layer_idx: int, total_layers: int) -> dict:
        if layer_idx in self._gpu_cache:
            self._gpu_cache.move_to_end(layer_idx)
            return self._gpu_cache[layer_idx]

        for next_idx in range(layer_idx + 1, min(total_layers, layer_idx + 1 + self.prefetch_depth)):
            self._prefetch(next_idx)

        cpu_state = self._get_cpu_layer(layer_idx)
        gpu_state = self._to_gpu(cpu_state)
        self._gpu_cache[layer_idx] = gpu_state
        self._gpu_cache.move_to_end(layer_idx)
        self._ensure_gpu_budget()
        return gpu_state

    def run_layerwise_inference(
        self,
        total_layers: int,
        compute_fn,
    ) -> list:
        outputs = []
        for layer_idx in range(total_layers):
            layer_state = self.load_layer(layer_idx, total_layers)
            outputs.append(compute_fn(layer_idx, layer_state))
            self._unload_from_gpu(layer_idx)
        return outputs
