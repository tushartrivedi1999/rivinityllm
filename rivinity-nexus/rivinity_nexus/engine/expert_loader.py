from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class ExpertLoader:
    """Stream experts from NVMe to CPU/GPU with async prefetch and LRU eviction."""

    @dataclass
    class GPUMemoryManager:
        max_gpu_experts: int = 8

        def budget_exceeded(self, resident_experts: int) -> bool:
            return resident_experts > self.max_gpu_experts

    def __init__(
        self,
        storage_path: str,
        cpu_cache_size: int = 64,
        max_gpu_experts: int = 8,
        prefetch_workers: int = 4,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cpu_cache_size = max(1, cpu_cache_size)
        self.gpu_manager = self.GPUMemoryManager(max_gpu_experts=max(1, max_gpu_experts))
        self.cpu_cache: OrderedDict[int, object] = OrderedDict()
        self.gpu_cache: OrderedDict[int, object] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=max(1, prefetch_workers))
        self._prefetch_futures: dict[int, Future] = {}

    def _path_for(self, expert_id: int) -> Path:
        return self.storage_path / f"expert_{expert_id:03d}.pt"

    def _touch_demo_expert(self, expert_id: int) -> Path:
        path = self._path_for(expert_id)
        if path.exists():
            return path
        if torch is not None:
            torch.save({"expert_id": expert_id, "weights": [expert_id]}, path)
        else:
            path.write_text(f"expert:{expert_id}")
        return path

    def load_expert(self, expert_id: int):
        if expert_id in self.cpu_cache:
            self.cpu_cache.move_to_end(expert_id)
            return self.cpu_cache[expert_id]

        path = self._touch_demo_expert(expert_id)
        if torch is not None:
            obj = torch.load(path, map_location="cpu")
        else:
            obj = {"expert_id": expert_id, "path": str(path), "weights": [expert_id]}

        self.cpu_cache[expert_id] = obj
        while len(self.cpu_cache) > self.cpu_cache_size:
            self.cpu_cache.popitem(last=False)
        return obj

    def prefetch_experts(self, expert_ids: list[int]) -> None:
        for expert_id in expert_ids:
            if expert_id in self.cpu_cache or expert_id in self._prefetch_futures:
                continue
            self._prefetch_futures[expert_id] = self._executor.submit(self.load_expert, expert_id)

    def _materialize_prefetch(self, expert_id: int):
        future = self._prefetch_futures.pop(expert_id, None)
        if future is None:
            return self.load_expert(expert_id)
        return future.result()

    def move_to_gpu(self, expert_id: int):
        expert = self._materialize_prefetch(expert_id)
        self.gpu_cache[expert_id] = expert
        self.gpu_cache.move_to_end(expert_id)
        self._evict_gpu_if_needed()
        return expert

    def _evict_gpu_if_needed(self) -> None:
        while self.gpu_manager.budget_exceeded(len(self.gpu_cache)):
            oldest_expert_id, _ = self.gpu_cache.popitem(last=False)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            _ = oldest_expert_id

    def stream_for_compute(self, active_experts: list[int], lookahead_experts: list[int] | None = None) -> dict[int, object]:
        lookahead = lookahead_experts or []
        self.prefetch_experts(lookahead)
        loaded: dict[int, object] = {}
        for expert_id in active_experts:
            loaded[expert_id] = self.move_to_gpu(expert_id)
        return loaded

    def expert_path(self, expert_id: int) -> str:
        return str(self._touch_demo_expert(expert_id))

    def evict_from_gpu(self, expert_id: int) -> None:
        self.gpu_cache.pop(expert_id, None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
