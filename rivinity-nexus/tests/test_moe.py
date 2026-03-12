from rivinity_nexus.engine.moe import (
    DistributedTokenRouter,
    DynamicExpertLoader,
    ExpertShardPlanner,
    KVCacheManager,
    MoEArchitecture,
    SparseExpertRouter,
)


def test_moe_memory_estimates() -> None:
    moe = MoEArchitecture(total_params_trillion=10.0, num_experts=256, top_k_experts=2, active_params_billion=40.0, quantization_bits=4)
    assert moe.dense_fp16_memory_tb() == 20.0
    assert moe.dense_quantized_memory_tb() == 5.0
    assert moe.active_memory_gb() == 20.0
    assert moe.estimate_compute_reduction() == 128.0


def test_sparse_router_selects_top_k() -> None:
    router = SparseExpertRouter(num_experts=256, top_k=2)
    experts = router.select_experts("hello")
    assert len(experts) == 2
    assert all(0 <= e < 256 for e in experts)


def test_expert_sharding_distribution() -> None:
    planner = ExpertShardPlanner(num_experts=256, nodes=4)
    shards = planner.shard_map()
    assert shards["node-1"][0] == 0
    assert shards["node-4"][-1] == 255
    assert planner.locate_expert(103) == "node-2"


def test_distributed_token_routing_and_kv_cache() -> None:
    router = SparseExpertRouter(num_experts=256, top_k=2)
    sharder = ExpertShardPlanner(num_experts=256, nodes=4)
    dist = DistributedTokenRouter(router=router, sharder=sharder)
    route = dist.route_token("token-abc")
    assert len(route["experts"]) == 2
    assert len(route["nodes"]) == 2

    kv = KVCacheManager(quantization_bits=4, paged=True)
    desc = kv.describe()
    assert desc["compression_ratio_vs_fp16"] == 4.0
    assert desc["paged_kv_cache"] is True


def test_dynamic_expert_loading(tmp_path) -> None:
    loader = DynamicExpertLoader(root_dir=str(tmp_path / "experts"))
    meta = loader.load_path(14)
    assert meta["expert_id"] == 14
    assert meta["pipeline"] == "NVMe->RAM->GPU"
