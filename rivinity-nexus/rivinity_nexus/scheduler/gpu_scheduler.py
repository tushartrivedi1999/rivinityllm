from dataclasses import dataclass
from enum import Enum
import logging

import httpx
from sqlalchemy.orm import Session

from rivinity_nexus.config.settings import get_settings
from rivinity_nexus.core.resilience import retry_async_call
from rivinity_nexus.models.entities import GpuVendorNode


class SchedulingStrategy(str, Enum):
    least_loaded = "least_loaded"
    vram_aware = "vram_aware"
    cost_aware = "cost_aware"


@dataclass
class GpuNode:
    vendor_name: str
    node_id: str
    node_ip: str
    gpu_type: str
    vram: int
    total_gpus: int
    free_gpus: int
    total_vram_gb: int
    used_vram_gb: int
    cost_per_hour: float
    region: str

    @property
    def load_ratio(self) -> float:
        if self.total_gpus == 0:
            return 1.0
        return (self.total_gpus - self.free_gpus) / self.total_gpus

    @property
    def free_vram_gb(self) -> int:
        return max(0, self.total_vram_gb - self.used_vram_gb)


@dataclass
class NodeAllocation:
    vendor_name: str
    node_id: str
    node_ip: str
    gpu_type: str
    vram: int
    allocated_gpus: int
    estimated_cost_per_hour: float
    free_vram_gb_after: int


@dataclass
class AllocationPlan:
    strategy: SchedulingStrategy
    allocations: list[NodeAllocation]
    total_allocated_gpus: int
    estimated_total_cost_per_hour: float


class GpuScheduler:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logging.getLogger("rivinity_nexus.gpu_scheduler")

    async def detect_gpu_nodes(self, db: Session | None = None) -> list[GpuNode]:
        if db is not None:
            rows = db.query(GpuVendorNode).filter(GpuVendorNode.is_active.is_(True)).all()
            if rows:
                return [
                    GpuNode(
                        vendor_name=row.vendor_name,
                        node_id=f"vendor-node-{row.id}",
                        node_ip=row.node_ip,
                        gpu_type=row.gpu_type,
                        vram=row.vram,
                        total_gpus=row.total_gpus,
                        free_gpus=row.available_gpus,
                        total_vram_gb=row.vram,
                        used_vram_gb=0,
                        cost_per_hour=float(row.price_per_hour),
                        region="vendor",
                    )
                    for row in rows
                ]

        headers = {"Authorization": f"Bearer {self.settings.gpu_vendor_api_token}"}
        async with httpx.AsyncClient(base_url=self.settings.gpu_vendor_api_base, timeout=20.0) as client:
            try:
                response = await retry_async_call(
                    lambda: client.get("/nodes/metrics", headers=headers),
                    attempts=3,
                    delay_seconds=0.2,
                    op_name="gpu_nodes_metrics",
                )
                response.raise_for_status()
                data = response.json()
                return [
                    GpuNode(
                        vendor_name=item.get("vendor", "external"),
                        node_id=item["node_id"],
                        node_ip=item.get("node_ip", "0.0.0.0"),
                        gpu_type=item["gpu_type"],
                        vram=int(item.get("total_vram_gb", 0)),
                        total_gpus=item["total_gpus"],
                        free_gpus=item["free_gpus"],
                        total_vram_gb=item["total_vram_gb"],
                        used_vram_gb=item["used_vram_gb"],
                        cost_per_hour=float(item.get("cost_per_hour", 1.0)),
                        region=item.get("region", "unknown"),
                    )
                    for item in data
                ]
            except Exception as exc:
                self.logger.warning("gpu_metrics_endpoint_failed", extra={"error": str(exc)})
                fallback = await retry_async_call(
                    lambda: client.get("/availability", headers=headers),
                    attempts=3,
                    delay_seconds=0.2,
                    op_name="gpu_availability_fallback",
                )
                fallback.raise_for_status()
                data = fallback.json()
                nodes: list[GpuNode] = []
                for idx, item in enumerate(data):
                    available = int(item.get("available", 0))
                    total_gpus = max(available, int(item.get("total", available or 1)))
                    nodes.append(
                        GpuNode(
                            vendor_name=item.get("vendor", "external"),
                            node_id=item.get("node_id", f"node-{idx}"),
                            node_ip=item.get("node_ip", "0.0.0.0"),
                            gpu_type=item.get("gpu_type", "unknown"),
                            vram=int(item.get("total_vram_gb", total_gpus * 24)),
                            total_gpus=total_gpus,
                            free_gpus=available,
                            total_vram_gb=int(item.get("total_vram_gb", total_gpus * 24)),
                            used_vram_gb=int(item.get("used_vram_gb", 0)),
                            cost_per_hour=float(item.get("cost_per_hour", 1.0)),
                            region=item.get("region", "unknown"),
                        )
                    )
                return nodes

    def _sort_nodes(self, nodes: list[GpuNode], strategy: SchedulingStrategy) -> list[GpuNode]:
        if strategy == SchedulingStrategy.least_loaded:
            return sorted(nodes, key=lambda n: (n.load_ratio, -n.free_gpus))
        if strategy == SchedulingStrategy.vram_aware:
            return sorted(nodes, key=lambda n: (-n.free_vram_gb, n.load_ratio))
        return sorted(nodes, key=lambda n: (n.cost_per_hour, n.load_ratio, -n.free_gpus))

    def build_allocation_plan(
        self,
        nodes: list[GpuNode],
        strategy: SchedulingStrategy,
        required_gpus: int,
        required_vram_gb: int,
        gpu_type: str | None = None,
    ) -> AllocationPlan:
        filtered = [n for n in nodes if n.free_gpus > 0 and n.free_vram_gb >= required_vram_gb]
        if gpu_type:
            filtered = [n for n in filtered if n.gpu_type == gpu_type]

        ordered = self._sort_nodes(filtered, strategy)

        remaining = required_gpus
        allocations: list[NodeAllocation] = []
        total_cost = 0.0

        for node in ordered:
            if remaining <= 0:
                break
            allocate_count = min(node.free_gpus, remaining)
            if allocate_count <= 0:
                continue
            remaining -= allocate_count
            cost = node.cost_per_hour * allocate_count
            total_cost += cost

            vram_after = max(0, node.free_vram_gb - required_vram_gb)
            allocations.append(
                NodeAllocation(
                    vendor_name=node.vendor_name,
                    node_id=node.node_id,
                    node_ip=node.node_ip,
                    gpu_type=node.gpu_type,
                    vram=node.vram,
                    allocated_gpus=allocate_count,
                    estimated_cost_per_hour=round(cost, 4),
                    free_vram_gb_after=vram_after,
                )
            )

        return AllocationPlan(
            strategy=strategy,
            allocations=allocations,
            total_allocated_gpus=sum(a.allocated_gpus for a in allocations),
            estimated_total_cost_per_hour=round(total_cost, 4),
        )
