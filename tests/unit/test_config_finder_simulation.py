"""Unit tests for simulation integration in ConfigFinder."""

import pytest
from unittest.mock import MagicMock
from planner.simulation.client import SimulationResult


def _sim_result(ttft=26.0, itl=10.0, e2e=3000.0, kv_fail=0.0, preempt=0.0):
    return SimulationResult(
        ttft_p95_ms=ttft, itl_p95_ms=itl, e2e_p95_ms=e2e,
        kv_allocation_failure_rate=kv_fail, preemption_rate=preempt,
        responses_per_sec=9.0, source="simulation",
    )


@pytest.mark.unit
def test_simulation_result_overwrites_db_latency():
    """When simulation succeeds, its latency replaces DB benchmark latency."""
    from planner.recommendation.config_finder import ConfigFinder

    sim_client = MagicMock()
    sim_client.is_available.return_value = True
    sim_client.simulate.return_value = _sim_result(ttft=20.0, e2e=2000.0)

    planner = ConfigFinder.__new__(ConfigFinder)
    from planner.simulation.cache import SimulationCache
    planner._sim_cache = SimulationCache(max_size=10)
    result = planner._get_latency(sim_client, "model", "H100", 1, 512, 256, 9.0, db_ttft=100, db_e2e=9000)
    assert result.ttft_p95_ms == pytest.approx(20.0)
    assert result.source == "simulation"


@pytest.mark.unit
def test_db_latency_used_when_simulation_unavailable():
    """When simulation returns None, DB latency is preserved."""
    from planner.recommendation.config_finder import ConfigFinder

    sim_client = MagicMock()
    sim_client.simulate.return_value = None

    planner = ConfigFinder.__new__(ConfigFinder)
    from planner.simulation.cache import SimulationCache
    planner._sim_cache = SimulationCache(max_size=10)
    result = planner._get_latency(sim_client, "model", "H100", 1, 512, 256, 9.0, db_ttft=100, db_e2e=9000)
    assert result.ttft_p95_ms == pytest.approx(100.0)
    assert result.source == "database"
