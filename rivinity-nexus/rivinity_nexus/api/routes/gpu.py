from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from rivinity_nexus.api.deps import get_current_user, get_db
from rivinity_nexus.data.schemas import (
    GpuAllocationPlanResponse,
    GpuAvailabilityResponse,
    GpuNodeAllocationResponse,
    GpuScheduleRequest,
    GpuVendorNodeRegisterRequest,
    GpuVendorNodeResponse,
)
from rivinity_nexus.models.entities import GpuVendorNode, User
from rivinity_nexus.scheduler.gpu_allocator import VendorGpuAllocator
from rivinity_nexus.scheduler.gpu_scheduler import GpuScheduler, SchedulingStrategy
from rivinity_nexus.monitoring.metrics import record_gpu_node_metrics

router = APIRouter(prefix="/gpu", tags=["gpu"])
allocator = VendorGpuAllocator()
scheduler = GpuScheduler()


class GpuAllocationRequest(BaseModel):
    gpu_type: str
    hours: int = Field(default=1, ge=1, le=168)


@router.post("/vendors/register", response_model=GpuVendorNodeResponse, status_code=status.HTTP_201_CREATED)
def register_vendor_node(
    payload: GpuVendorNodeRegisterRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> GpuVendorNodeResponse:
    existing = db.query(GpuVendorNode).filter(GpuVendorNode.node_ip == payload.node_ip).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Node already registered")

    node = GpuVendorNode(
        vendor_name=payload.vendor_name,
        node_ip=payload.node_ip,
        gpu_type=payload.gpu_type,
        vram=payload.vram,
        price_per_hour=payload.price_per_hour,
        total_gpus=payload.total_gpus,
        available_gpus=min(payload.available_gpus, payload.total_gpus),
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return GpuVendorNodeResponse(
        id=node.id,
        vendor_name=node.vendor_name,
        node_ip=node.node_ip,
        gpu_type=node.gpu_type,
        vram=node.vram,
        price_per_hour=float(node.price_per_hour),
        total_gpus=node.total_gpus,
        available_gpus=node.available_gpus,
    )


@router.get("/vendors", response_model=list[GpuVendorNodeResponse])
def list_vendor_nodes(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[GpuVendorNodeResponse]:
    rows = db.query(GpuVendorNode).filter(GpuVendorNode.is_active.is_(True)).all()
    return [
        GpuVendorNodeResponse(
            id=row.id,
            vendor_name=row.vendor_name,
            node_ip=row.node_ip,
            gpu_type=row.gpu_type,
            vram=row.vram,
            price_per_hour=float(row.price_per_hour),
            total_gpus=row.total_gpus,
            available_gpus=row.available_gpus,
        )
        for row in rows
    ]


@router.get("/available", response_model=list[GpuAvailabilityResponse])
async def available_gpu(db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> list[GpuAvailabilityResponse]:
    nodes = db.query(GpuVendorNode).filter(GpuVendorNode.is_active.is_(True)).all()
    if nodes:
        by_vendor_type: dict[tuple[str, str], int] = {}
        for node in nodes:
            key = (node.vendor_name, node.gpu_type)
            by_vendor_type[key] = by_vendor_type.get(key, 0) + node.available_gpus
        return [
            GpuAvailabilityResponse(vendor=vendor, gpu_type=gpu_type, available=available, region="vendor")
            for (vendor, gpu_type), available in by_vendor_type.items()
        ]

    try:
        availability = await allocator.available()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="GPU availability lookup failed") from exc
    return [GpuAvailabilityResponse(**item.__dict__) for item in availability]


@router.post("/schedule", response_model=GpuAllocationPlanResponse)
async def schedule_gpu(
    payload: GpuScheduleRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> GpuAllocationPlanResponse:
    try:
        nodes = await scheduler.detect_gpu_nodes(db=db)
        for node in nodes:
            record_gpu_node_metrics(
                vendor=node.vendor_name,
                node_id=node.node_id,
                gpu_type=node.gpu_type,
                total_gpus=node.total_gpus,
                free_gpus=node.free_gpus,
                total_vram_gb=node.total_vram_gb,
                used_vram_gb=node.used_vram_gb,
            )
        plan = scheduler.build_allocation_plan(
            nodes=nodes,
            strategy=SchedulingStrategy(payload.strategy),
            required_gpus=payload.required_gpus,
            required_vram_gb=payload.required_vram_gb,
            gpu_type=payload.gpu_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="GPU scheduling failed") from exc

    if plan.total_allocated_gpus < payload.required_gpus:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Insufficient GPU capacity: allocated={plan.total_allocated_gpus}, required={payload.required_gpus}",
        )

    return GpuAllocationPlanResponse(
        strategy=plan.strategy.value,
        allocations=[GpuNodeAllocationResponse(**item.__dict__) for item in plan.allocations],
        total_allocated_gpus=plan.total_allocated_gpus,
        estimated_total_cost_per_hour=plan.estimated_total_cost_per_hour,
    )


@router.post("/allocate")
async def allocate_gpu(
    payload: GpuAllocationRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict[str, str]:
    node = (
        db.query(GpuVendorNode)
        .filter(
            GpuVendorNode.is_active.is_(True),
            GpuVendorNode.gpu_type == payload.gpu_type,
            GpuVendorNode.available_gpus > 0,
        )
        .order_by(GpuVendorNode.price_per_hour.asc())
        .first()
    )
    if node:
        node.available_gpus -= 1
        db.commit()
        return {"node_id": f"vendor-node-{node.id}", "lease_id": f"local-{node.id}-{payload.hours}", "gpu_type": node.gpu_type}

    try:
        lease = await allocator.allocate(payload.gpu_type, payload.hours)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="GPU vendor allocation failed") from exc
    return {"node_id": lease.node_id, "lease_id": lease.lease_id, "gpu_type": lease.gpu_type}
