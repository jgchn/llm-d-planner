from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from planner.simulation.client import SimulationClient

logger = logging.getLogger(__name__)

_IMPROVEMENT_THRESHOLD_PCT = 5.0


@dataclass
class CacheAffinityRecommendation:
    policy: Literal["cache_aware", "round_robin"]
    reasoning: str
    simulated_ttft_improvement_pct: float


def recommend_cache_affinity(
    client: SimulationClient,
    model: str,
    gpu: str,
    tp: int,
    prompt_tokens: int,
    output_tokens: int,
    qps: float,
    system_prompt_tokens: int,
) -> CacheAffinityRecommendation:
    if system_prompt_tokens <= 0:
        return CacheAffinityRecommendation(
            policy="round_robin",
            reasoning="No shared system prompt — prefix caching provides no benefit.",
            simulated_ttft_improvement_pct=0.0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_base = executor.submit(
            client.simulate, model, gpu, tp, prompt_tokens, output_tokens, qps,
            prefix_tokens=0,
        )
        f_warm = executor.submit(
            client.simulate, model, gpu, tp, prompt_tokens, output_tokens, qps,
            prefix_tokens=system_prompt_tokens,
        )
        base = f_base.result()
        warm = f_warm.result()

    if base is None or warm is None or base.ttft_p95_ms <= 0:
        return CacheAffinityRecommendation(
            policy="round_robin",
            reasoning="Cache affinity simulation unavailable — defaulting to round-robin.",
            simulated_ttft_improvement_pct=0.0,
        )

    improvement = (base.ttft_p95_ms - warm.ttft_p95_ms) / base.ttft_p95_ms * 100

    if improvement >= _IMPROVEMENT_THRESHOLD_PCT:
        return CacheAffinityRecommendation(
            policy="cache_aware",
            reasoning=(
                f"Routing requests to instances with your {system_prompt_tokens}-token "
                f"system prompt cached reduces TTFT p95 by {improvement:.0f}%. "
                f"Use prefix-affinity scheduling (e.g., llm-d kv-aware routing)."
            ),
            simulated_ttft_improvement_pct=round(improvement, 1),
        )
    return CacheAffinityRecommendation(
        policy="round_robin",
        reasoning=(
            f"System prompt caching gives <{_IMPROVEMENT_THRESHOLD_PCT:.0f}% TTFT gain "
            f"({improvement:.0f}%) — round-robin is sufficient."
        ),
        simulated_ttft_improvement_pct=round(improvement, 1),
    )
