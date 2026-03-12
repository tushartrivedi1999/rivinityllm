import asyncio

from rivinity_nexus.distributed.expert_rpc import ExpertRPCClient
from rivinity_nexus.engine.expert_loader import ExpertLoader
from rivinity_nexus.engine.kv_cache_manager import KVCacheManager
from rivinity_nexus.engine.layer_streamer import LayerStreamer
from rivinity_nexus.engine.moe_router import MoERouter
from rivinity_nexus.scheduler.expert_scheduler import ExpertScheduler


def test_moe_router_topk() -> None:
    router = MoERouter(num_experts=16, top_k=2)
    out = router.route([1.0, 2.0, 3.0])
    assert len(out.expert_ids) == 2
    assert len(out.scores) == 2


def test_expert_loader_and_cache(tmp_path) -> None:
    loader = ExpertLoader(str(tmp_path / "experts"), cpu_cache_size=2, max_gpu_experts=1)
    e1 = loader.load_expert(1)
    e2 = loader.load_expert(2)
    assert e1 is not None and e2 is not None
    loader.load_expert(3)
    assert len(loader.cpu_cache) <= 2
    loader.prefetch_experts([1, 2])
    loader.stream_for_compute([1, 2])
    assert len(loader.gpu_cache) == 1


def test_kv_cache_paging() -> None:
    kv = KVCacheManager(max_gpu_entries=1, quantization_bits=4, paged=True)
    kv.put("a", "A")
    kv.put("b", "B")
    assert kv.get("a") == "A"
    desc = kv.describe()
    assert desc["quantization_bits"] == 4


def test_layer_streamer_pipeline() -> None:
    ls = LayerStreamer()
    assert "NVMe" in ls.describe()
    assert "GPU execution" in ls.describe()


def test_expert_scheduler_assign() -> None:
    sch = ExpertScheduler()
    mapping = sch.assign([1, 2, 3], ["node-1", "node-2"])
    assert mapping[1] == "node-1"
    assert mapping[2] == "node-2"


def test_expert_rpc_async() -> None:
    rpc = ExpertRPCClient()

    async def _run():
        res = await rpc.gather([("node-1", ["tok"], 7)])
        assert res[0]["expert_id"] == 7

    asyncio.run(_run())


def test_expert_rpc_distributed_batching() -> None:
    rpc = ExpertRPCClient(batch_size=2)

    async def _run():
        out = await rpc.execute_distributed({7: "node-a", 8: "node-b"}, ["t1", "t2", "t3"])
        assert 7 in out and 8 in out
        assert len(out[7]) == 3

    asyncio.run(_run())
