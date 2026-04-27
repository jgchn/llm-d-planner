"""Unit tests for POST /api/v1/explore-config endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from planner.api.app import create_app
from planner.simulation.client import SimulationResult


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def _sim_result():
    return SimulationResult(
        ttft_p95_ms=26.3, itl_p95_ms=10.7, e2e_p95_ms=3856.1,
        kv_allocation_failure_rate=0.0, preemption_rate=0.0,
        responses_per_sec=9.91, source="simulation",
    )


@pytest.mark.unit
def test_explore_config_returns_simulation_metrics(client):
    with patch("planner.api.routes.explore.SimulationClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.simulate.return_value = _sim_result()
        MockClient.return_value = mock_instance

        response = client.post("/api/v1/explore-config", json={
            "model": "meta-llama/llama-3.1-8b-instruct",
            "gpu_type": "H100",
            "tensor_parallelism": 1,
            "gpu_count": 1,
            "prompt_tokens": 512,
            "output_tokens": 256,
            "qps": 9.0,
            "replicas": 1,
        })

    assert response.status_code == 200
    data = response.json()
    assert data["simulation"]["ttft_p95_ms"] == pytest.approx(26.3)
    assert "monthly_cost_usd" in data
    assert "reliability_status" in data


@pytest.mark.unit
def test_explore_config_returns_503_when_binary_unavailable(client):
    with patch("planner.api.routes.explore.SimulationClient") as MockClient, \
         patch("planner.api.routes.explore.get_default_cache") as MockCache:
        mock_instance = MagicMock()
        mock_instance.simulate.return_value = None
        mock_instance.is_available.return_value = False
        MockClient.return_value = mock_instance
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        MockCache.return_value = mock_cache

        response = client.post("/api/v1/explore-config", json={
            "model": "meta-llama/llama-3.1-8b-instruct",
            "gpu_type": "H100",
            "tensor_parallelism": 1,
            "gpu_count": 1,
            "prompt_tokens": 512,
            "output_tokens": 256,
            "qps": 9.0,
            "replicas": 1,
        })

    assert response.status_code == 503


@pytest.mark.unit
def test_explore_config_returns_400_for_zero_qps(client):
    response = client.post("/api/v1/explore-config", json={
        "model": "m", "gpu_type": "H100", "tensor_parallelism": 1,
        "gpu_count": 1, "prompt_tokens": 512, "output_tokens": 256,
        "qps": 0.0, "replicas": 1,
    })
    assert response.status_code == 422
