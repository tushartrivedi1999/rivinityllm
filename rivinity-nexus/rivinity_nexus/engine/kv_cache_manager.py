class KVCacheManager:
    """Paged KV cache with optional CPU offload simulation."""

    def __init__(self, max_gpu_entries: int = 256, quantization_bits: int = 4, paged: bool = True) -> None:
        self.max_gpu_entries = max(1, max_gpu_entries)
        self.quantization_bits = quantization_bits
        self.paged = paged
        self.gpu_cache: dict[str, str] = {}
        self.cpu_cache: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        if len(self.gpu_cache) >= self.max_gpu_entries:
            # Offload oldest item to CPU cache.
            oldest = next(iter(self.gpu_cache))
            self.cpu_cache[oldest] = self.gpu_cache.pop(oldest)
        self.gpu_cache[key] = value

    def get(self, key: str) -> str | None:
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        if key in self.cpu_cache:
            val = self.cpu_cache.pop(key)
            self.put(key, val)
            return val
        return None

    def describe(self) -> dict:
        return {
            "quantization_bits": self.quantization_bits,
            "paged_kv_cache": self.paged,
            "gpu_entries": len(self.gpu_cache),
            "cpu_offloaded_entries": len(self.cpu_cache),
        }
