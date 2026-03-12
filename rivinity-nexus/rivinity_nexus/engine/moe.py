from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass
class MoEArchitecture:
    total_params_trillion: float = 10.0
    num_experts: int = 256
    top_k_experts: int = 2
    active_params_billion: float = 40.0
    quantization_bits: int = 4

    def dense_fp16_memory_tb(self) -> float:
        return round(self.total_params_trillion * 2.0, 2)

    def dense_quantized_memory_tb(self) -> float:
        return round(self.total_params_trillion * (self.quantization_bits / 8.0), 2)

    def active_memory_gb(self) -> float:
        bytes_per_param = self.quantization_bits / 8.0
        gb = self.active_params_billion * 1e9 * bytes_per_param / 1e9
        return round(gb, 2)

    def estimate_compute_reduction(self) -> float:
        if self.top_k_experts <= 0:
            return 1.0
        return round(max(1.0, self.num_experts / self.top_k_experts), 2)


class SparseExpertRouter:
    """Deterministic token router for sparse MoE simulation."""

    def __init__(self, num_experts: int = 256, top_k: int = 2) -> None:
        self.num_experts = num_experts
        self.top_k = top_k

    def select_experts(self, token: str) -> list[int]:
        if self.num_experts <= 0:
            return []
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        base = int(digest[:8], 16)
        selected: list[int] = []
        for i in range(self.top_k):
            idx = (base + i * 7919) % self.num_experts
            selected.append(idx)
        return selected


class ExpertShardPlanner:
    """Map experts to nodes for distributed expert sharding."""

    def __init__(self, num_experts: int = 256, nodes: int = 4) -> None:
        self.num_experts = num_experts
        self.nodes = max(1, nodes)

    def shard_map(self) -> dict[str, list[int]]:
        experts_per_node = max(1, self.num_experts // self.nodes)
        mapping: dict[str, list[int]] = {}
        for node_idx in range(self.nodes):
            start = node_idx * experts_per_node
            end = self.num_experts if node_idx == self.nodes - 1 else start + experts_per_node
            mapping[f"node-{node_idx + 1}"] = list(range(start, min(end, self.num_experts)))
        return mapping

    def locate_expert(self, expert_id: int) -> str:
        for node, experts in self.shard_map().items():
            if expert_id in experts:
                return node
        return "node-unknown"


class DistributedTokenRouter:
    """Resolve token -> experts -> nodes for distributed routing."""

    def __init__(self, router: SparseExpertRouter, sharder: ExpertShardPlanner) -> None:
        self.router = router
        self.sharder = sharder

    def route_token(self, token: str) -> dict:
        experts = self.router.select_experts(token)
        node_routes = [self.sharder.locate_expert(e) for e in experts]
        return {"token": token, "experts": experts, "nodes": node_routes}


class KVCacheManager:
    """Simulate quantized/paged KV-cache strategies."""

    def __init__(self, quantization_bits: int = 8, paged: bool = True) -> None:
        self.quantization_bits = quantization_bits
        self.paged = paged

    def compression_ratio(self) -> float:
        return round(16 / max(1, self.quantization_bits), 2)

    def describe(self) -> dict:
        return {
            "quantization_bits": self.quantization_bits,
            "compression_ratio_vs_fp16": self.compression_ratio(),
            "paged_kv_cache": self.paged,
        }


class DynamicExpertLoader:
    """Simulate NVMe -> RAM -> GPU expert loading path."""

    def __init__(self, root_dir: str = "/tmp/rivinity/experts") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_expert(self, expert_id: int) -> str:
        path = self.root / f"expert_{expert_id}.bin"
        if not path.exists():
            path.write_text(f"expert:{expert_id}")
        return str(path)

    def load_path(self, expert_id: int) -> dict:
        file_path = self.ensure_expert(expert_id)
        return {
            "expert_id": expert_id,
            "path": file_path,
            "pipeline": "NVMe->RAM->GPU",
        }
