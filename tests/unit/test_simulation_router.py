import pytest
from unittest.mock import MagicMock
from planner.simulation.router import recommend_cache_affinity, CacheAffinityRecommendation
from planner.simulation.client import SimulationResult


def _result(ttft: float) -> SimulationResult:
    return SimulationResult(
        ttft_p95_ms=ttft, itl_p95_ms=10.0, e2e_p95_ms=1000.0,
        kv_allocation_failure_rate=0.0, preemption_rate=0.0,
        responses_per_sec=9.0,
    )


@pytest.mark.unit
def test_returns_round_robin_when_no_system_prompt():
    client = MagicMock()
    rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=0)
    assert rec.policy == "round_robin"
    assert rec.simulated_ttft_improvement_pct == 0.0
    client.simulate.assert_not_called()


@pytest.mark.unit
def test_returns_cache_aware_when_improvement_exceeds_threshold():
    client = MagicMock()
    # baseline (no prefix): 100ms TTFT; warm (with prefix): 70ms TTFT → 30% improvement
    client.simulate.side_effect = [_result(100.0), _result(70.0)]
    rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=400)
    assert rec.policy == "cache_aware"
    assert rec.simulated_ttft_improvement_pct == pytest.approx(30.0, abs=0.1)


@pytest.mark.unit
def test_returns_round_robin_when_improvement_below_threshold():
    client = MagicMock()
    # only 2% improvement — below 5% threshold
    client.simulate.side_effect = [_result(100.0), _result(98.0)]
    rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=400)
    assert rec.policy == "round_robin"


@pytest.mark.unit
def test_returns_round_robin_when_simulation_unavailable():
    client = MagicMock()
    client.simulate.return_value = None
    rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=400)
    assert rec.policy == "round_robin"
    assert rec.simulated_ttft_improvement_pct == 0.0


@pytest.mark.unit
def test_runs_two_simulations_with_correct_prefix_tokens():
    client = MagicMock()
    client.simulate.side_effect = [_result(100.0), _result(70.0)]
    recommend_cache_affinity(client, "my-model", "A100", 2, 1024, 512, 12.0, system_prompt_tokens=600)
    calls = client.simulate.call_args_list
    assert len(calls) == 2
    # first call: prefix_tokens=0 (baseline)
    assert calls[0].kwargs.get("prefix_tokens", calls[0].args[-1] if calls[0].args else None) == 0 or \
           calls[0].args[6] == 0
    # second call: prefix_tokens=600
    assert 600 in calls[1].args or calls[1].kwargs.get("prefix_tokens") == 600
