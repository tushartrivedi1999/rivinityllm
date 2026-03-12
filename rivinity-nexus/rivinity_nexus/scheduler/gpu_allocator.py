from dataclasses import dataclass

import httpx

from rivinity_nexus.config.settings import get_settings


@dataclass
class GpuAllocation:
    node_id: str
    gpu_type: str
    lease_id: str


@dataclass
class GpuAvailability:
    vendor: str
    gpu_type: str
    available: int
    region: str


class VendorGpuAllocator:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def allocate(self, gpu_type: str, hours: int = 1) -> GpuAllocation:
        payload = {"gpu_type": gpu_type, "hours": hours}
        headers = {"Authorization": f"Bearer {self.settings.gpu_vendor_api_token}"}
        async with httpx.AsyncClient(base_url=self.settings.gpu_vendor_api_base, timeout=30.0) as client:
            response = await client.post("/leases", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return GpuAllocation(node_id=data["node_id"], gpu_type=data["gpu_type"], lease_id=data["lease_id"])

    async def available(self) -> list[GpuAvailability]:
        headers = {"Authorization": f"Bearer {self.settings.gpu_vendor_api_token}"}
        async with httpx.AsyncClient(base_url=self.settings.gpu_vendor_api_base, timeout=15.0) as client:
            response = await client.get("/availability", headers=headers)
            response.raise_for_status()
            data = response.json()

        return [
            GpuAvailability(
                vendor=item["vendor"],
                gpu_type=item["gpu_type"],
                available=item["available"],
                region=item["region"],
            )
            for item in data
        ]
