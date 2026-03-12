from rivinity_nexus.core.resilience import retry_call


def test_retry_call_recovers_after_transient_error() -> None:
    state = {"count": 0}

    def flaky() -> str:
        state["count"] += 1
        if state["count"] < 2:
            raise RuntimeError("transient")
        return "ok"

    result = retry_call(flaky, attempts=3, delay_seconds=0.0, op_name="test")
    assert result == "ok"
    assert state["count"] == 2
