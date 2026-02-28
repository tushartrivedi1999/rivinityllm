import pytest

from rivinityllm_tune.inference.planner import (
    PlanError,
    build_inference_plan,
    estimate_weight_footprint_gb,
)


def test_weight_footprint_estimation_for_1000b_4bit() -> None:
    weight_gb = estimate_weight_footprint_gb(parameter_count_b=1000, quantization_bits=4)
    assert weight_gb > 400


def test_1000b_plan_uses_hierarchical_strategy() -> None:
    plan = build_inference_plan(
        parameter_count_b=1000,
        quantization_bits=4,
        target_gpu_vram_gb=8,
        sequence_length=8192,
        batch_size=2,
    )
    assert plan.strategy == "hierarchical_offload+tp+pp+speculative_prefetch"
    assert plan.gpu_count >= 1
    assert plan.pipeline_parallelism >= 2
    assert plan.nvme_offload_gb > plan.cpu_cache_gb


def test_invalid_quantization_raises() -> None:
    with pytest.raises(PlanError):
        estimate_weight_footprint_gb(parameter_count_b=70, quantization_bits=5)
