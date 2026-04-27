"""Explore-config endpoint: run inference-sim for scenario analysis."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from planner.simulation.cache import get_default_cache
from planner.simulation.client import SimulationClient
from planner.simulation.router import CacheAffinityRecommendation, recommend_cache_affinity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_GPU_HOURLY_RATES: dict[str, float] = {
    "H100": 8.0, "A100-80GB": 4.5, "A100-40GB": 3.0,
    "L4": 0.7, "L40S": 2.0, "T4": 0.35,
}
_HOURS_PER_MONTH = 730


class ExploreConfigRequest(BaseModel):
    model: str
    gpu_type: str
    tensor_parallelism: int = Field(ge=1)
    gpu_count: int = Field(ge=1)
    prompt_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    qps: float = Field(gt=0, description="Must be > 0")
    replicas: int = Field(ge=1)
    prefix_tokens: int = Field(default=0, ge=0)


class ExploreConfigResponse(BaseModel):
    simulation: dict
    monthly_cost_usd: float
    reliability_status: Literal["ok", "warning", "critical"]
    cache_affinity_recommendation: CacheAffinityRecommendation | None


def _reliability_status(
    kv_rate: float, preempt_rate: float
) -> Literal["ok", "warning", "critical"]:
    if kv_rate > 0.02:
        return "critical"
    if kv_rate > 0 or preempt_rate > 0.05:
        return "warning"
    return "ok"


@router.post("/explore-config", response_model=ExploreConfigResponse)
def explore_config(req: ExploreConfigRequest) -> ExploreConfigResponse:
    client = SimulationClient()
    cache = get_default_cache()

    cached = cache.get(
        req.model, req.gpu_type, req.tensor_parallelism,
        req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens,
    )
    sim = cached
    if sim is None:
        sim = client.simulate(
            req.model, req.gpu_type, req.tensor_parallelism,
            req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens,
        )
        if sim is not None:
            cache.put(
                req.model, req.gpu_type, req.tensor_parallelism,
                req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens, sim,
            )

    if sim is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation unavailable. Recommendation card values remain valid.",
        )

    hourly = _GPU_HOURLY_RATES.get(req.gpu_type, 2.0)
    monthly_cost = req.gpu_count * req.replicas * hourly * _HOURS_PER_MONTH

    affinity: CacheAffinityRecommendation | None = None
    if req.prefix_tokens > 0:
        affinity = recommend_cache_affinity(
            client, req.model, req.gpu_type, req.tensor_parallelism,
            req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens,
        )

    return ExploreConfigResponse(
        simulation={
            "ttft_p95_ms": sim.ttft_p95_ms,
            "itl_p95_ms": sim.itl_p95_ms,
            "e2e_p95_ms": sim.e2e_p95_ms,
            "kv_allocation_failure_rate": sim.kv_allocation_failure_rate,
            "preemption_rate": sim.preemption_rate,
            "responses_per_sec": sim.responses_per_sec,
        },
        monthly_cost_usd=round(monthly_cost, 2),
        reliability_status=_reliability_status(
            sim.kv_allocation_failure_rate, sim.preemption_rate
        ),
        cache_affinity_recommendation=affinity,
    )
