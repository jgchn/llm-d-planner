import pytest
from planner.simulation.cache import SimulationCache
from planner.simulation.client import SimulationResult


def _result(ttft: float) -> SimulationResult:
    return SimulationResult(
        ttft_p95_ms=ttft, itl_p95_ms=10.0, e2e_p95_ms=1000.0,
        kv_allocation_failure_rate=0.0, preemption_rate=0.0,
        responses_per_sec=9.0,
    )


@pytest.mark.unit
def test_cache_miss_returns_none():
    cache = SimulationCache(max_size=10)
    assert cache.get("model", "H100", 1, 512, 256, 9.0, 0) is None


@pytest.mark.unit
def test_cache_hit_returns_stored_result():
    cache = SimulationCache(max_size=10)
    result = _result(26.3)
    cache.put("model", "H100", 1, 512, 256, 9.0, 0, result)
    retrieved = cache.get("model", "H100", 1, 512, 256, 9.0, 0)
    assert retrieved is result


@pytest.mark.unit
def test_cache_evicts_oldest_entry_when_full():
    cache = SimulationCache(max_size=2)
    r1, r2, r3 = _result(1.0), _result(2.0), _result(3.0)
    cache.put("m", "H100", 1, 512, 256, 1.0, 0, r1)
    cache.put("m", "H100", 1, 512, 256, 2.0, 0, r2)
    cache.put("m", "H100", 1, 512, 256, 3.0, 0, r3)  # evicts r1
    assert cache.get("m", "H100", 1, 512, 256, 1.0, 0) is None
    assert cache.get("m", "H100", 1, 512, 256, 2.0, 0) is r2
    assert cache.get("m", "H100", 1, 512, 256, 3.0, 0) is r3


@pytest.mark.unit
def test_cache_key_includes_prefix_tokens():
    cache = SimulationCache(max_size=10)
    r0 = _result(26.0)
    r400 = _result(18.0)
    cache.put("m", "H100", 1, 512, 256, 9.0, 0, r0)
    cache.put("m", "H100", 1, 512, 256, 9.0, 400, r400)
    assert cache.get("m", "H100", 1, 512, 256, 9.0, 0) is r0
    assert cache.get("m", "H100", 1, 512, 256, 9.0, 400) is r400
