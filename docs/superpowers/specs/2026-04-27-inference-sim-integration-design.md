# Inference-Sim Integration Design

**Date**: 2026-04-27  
**Status**: Draft  
**Scope**: Prototype (Phase 1)

---

## Problem

The planner today estimates latency by looking up pre-computed benchmark data in PostgreSQL. This has two limitations:

1. **Only 4 fixed traffic profiles supported** (512→256, 1024→1024, 4096→512, 10240→1536 tokens). If a user's workload doesn't match, we snap to the nearest profile and accept the error silently.
2. **No cache affinity or reliability signal**. The planner cannot tell users whether their deployment will experience KV cache pressure, preemptions, or how cache-aware request scheduling would affect latency.

## Solution

Replace the static benchmark DB lookup with live calls to `inference-sim`, a Go discrete-event simulator that predicts latency for any (model, GPU, TP, traffic profile) combination in milliseconds. Add cache affinity comparison and reliability signals to every recommendation.

---

## Goals

- Accurate latency predictions for arbitrary traffic profiles, not just 4 fixed ones
- Per-recommendation cache affinity comparison with simulated (not estimated) improvement %
- Reliability signal (KV allocation failures, preemption rate) on every card
- Lightweight "Explore" panel per recommendation for stress-test scenario comparison
- Zero new services, zero new Docker containers for the prototype

## Non-Goals (prototype)

- True multi-instance routing simulation (requires a separate fork not used here)
- Queue-depth routing simulation (cannot be modeled with single-instance binary)
- Persistent scenario history across sessions
- Automatic model download or binary versioning
- Surfacing inference-sim predictions in simulator mode (KIND/vLLM-simulator deployments use DB benchmarks; inference-sim coefficients are calibrated for real GPU latency and are not applicable in simulator mode)

---

## Architecture

### What changes

**Before**: `config_finder.py` → SQL query on `exported_summaries` → p95 latency

**After**: `config_finder.py` → `SimulationClient.simulate()` → inference-sim subprocess → p95 latency  
**Fallback chain**: If binary unavailable or model not in `coefficients.yaml` → SQL query on `exported_summaries`. If SQL also has no data → log warning and exclude config from results (same as today's behavior for missing benchmark data).  
**Simulator mode**: If `DeploymentGenerator(simulator_mode=True)`, skip inference-sim entirely and use DB only. inference-sim coefficients are calibrated for real GPU latency; they are not meaningful for vLLM-simulator deployments.  
**Partial failures**: If simulation succeeds for some candidates and fails for others, continue with the results available. Each candidate independently falls back to DB. The `latency_source` field on each recommendation indicates which path was used.

```
User intent
    ↓
Traffic profile (prompt_tokens, output_tokens, qps, system_prompt_tokens)  ← system_prompt_tokens NEW
    ↓
config_finder.py  [plan_all_capacities()]
    ├─ Query DB for candidate (model, GPU, TP) combinations — same as today
    │   (this gives us the candidate list; we then re-derive latency via simulation)
    │
    ├─ Parallel simulation (ThreadPoolExecutor, max_workers=8)
    │   └─ For each candidate:
    │       SimulationClient.simulate(model, gpu, tp, prompt, output, qps)
    │       → on success: use SimulationResult for latency scoring
    │       → on None: fall back to DB benchmark latency for that candidate
    │   Note: executor queues work when candidates > 8; total time ≈ ceil(N/8) × 50ms
    │   Note: if expected_qps is None, default to 1.0 for simulation
    │
    ├─ Score all candidates using latency from simulation (or DB fallback)
    │   Scorer.score_latency() inputs are unchanged; only the source of
    │   ttft_p95/itl_p95/e2e_p95 values changes
    │
    ├─ Generate ranked lists via analyzer.py (unchanged)
    │
    └─ Cache affinity enrichment — post-processing step at end of plan_all_capacities()
        AFTER all configs are scored (balanced score is set on each DeploymentRecommendation)
        Sort all_configs by balanced_score descending, take top-3, run 6 parallel sims,
        then MUTATE those 3 recommendation objects in-place:
            rec.cache_affinity_recommendation = CacheAffinityRecommendation(...)
        generate_ranked_lists() is called by the caller on the already-enriched list.
        (ThreadPoolExecutor, 6 parallel simulations)
        ├─ Run A: prefix_tokens=0                    (baseline, no cache reuse)
        └─ Run B: prefix_tokens=system_prompt_tokens (cache-warm path)
            → CacheAffinityRecommendation with simulated_ttft_improvement_pct
        Skipped if system_prompt_tokens == 0 (no shared prefix to simulate) or simulator_mode is True.
        NOT skipped based on replica count — prefix caching benefit applies to single instances too.

New endpoint: POST /api/v1/explore-config
    → runs simulation with user-supplied overrides
    → returns SimulationResult + updated cost

UI: recommendation cards + Explore panel per card
```

---

## New Package: `src/planner/simulation/`

### `client.py`

Subprocess wrapper around the inference-sim binary.

**Binary location**: read from `INFERENCE_SIM_BIN` env var, default `../inference-sim/simulation_worker`.

**Makefile changes**: Add a `setup-inference-sim` target to the existing `Makefile`:
```makefile
setup-inference-sim:
	@if [ ! -f "$(INFERENCE_SIM_BIN)" ]; then \
		echo "Building inference-sim binary..."; \
		if ! command -v go &> /dev/null; then \
			echo "Error: Go toolchain not found. Install Go or set INFERENCE_SIM_BIN."; \
			exit 1; \
		fi; \
		cd $(dir $(INFERENCE_SIM_BIN)) && go build -o simulation_worker ./...; \
	fi
```
Call `setup-inference-sim` from within the existing `setup-backend` target (non-fatal: print a warning if it fails so users without Go can still run the planner with DB-only mode).

```python
@dataclass
class SimulationResult:
    ttft_p95_ms: float
    itl_p95_ms: float
    e2e_p95_ms: float
    kv_allocation_failure_rate: float   # failures / total_requests
    preemption_rate: float              # preemptions / total_requests
    responses_per_sec: float
    source: Literal["simulation", "database"]  # which path produced this result

def simulate(
    model: str,
    gpu: str,
    tp: int,
    prompt_tokens: int,
    output_tokens: int,
    qps: float,
    prefix_tokens: int = 0,
) -> SimulationResult | None:
    # Returns None if: binary unavailable, model not in coefficients.yaml, or qps <= 0
    # Caller falls back to DB on None
```

CLI args built from parameters, `--output json` flag added to inference-sim (small upstream PR).  
Timeout: 10 seconds per simulation call.

**JSON schema contract**: The `--output json` flag in inference-sim must emit a JSON object with exactly these keys (snake_case, milliseconds for latency):
```json
{
  "ttft_p95_ms": 26.3,
  "itl_p95_ms": 10.7,
  "e2e_p95_ms": 3856.1,
  "kv_allocation_failures": 0,
  "total_requests": 1000,
  "preemption_count": 3,
  "responses_per_sec": 9.91
}
```
`client.py` computes `kv_allocation_failure_rate = kv_allocation_failures / total_requests` and `preemption_rate = preemption_count / total_requests` from the raw counts. Validate this schema manually with a test invocation before building the Python client (step 1 of implementation order).

### `cache.py`

In-memory LRU cache (module-level singleton, persists across requests).  
Key: `(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)`  
Max size: 2048 entries — sized to cover both recommendation generation (~20 entries/request) and Explore panel exploration (~50 entries/session) without eviction under normal use.

**Scope**: Cross-request, shared across all users of the same server process. Results are deterministic (same inputs → same outputs from inference-sim), so sharing across users is safe and desirable. No cache invalidation needed; restart clears it.

### `router.py`

Runs cache affinity simulations for top-3 balanced candidates.

```python
@dataclass
class CacheAffinityRecommendation:
    policy: Literal["cache_aware", "round_robin"]
    reasoning: str
    simulated_ttft_improvement_pct: float   # positive = cache_aware is better
    # Note: this is a cache affinity comparison, not a network routing simulation.
    # Real cluster routing impact depends on replica count, cache size, and load balancer.

def recommend(
    config: DeploymentRecommendation,
    system_prompt_tokens: int,
) -> CacheAffinityRecommendation:
    # If system_prompt_tokens == 0: policy="round_robin", improvement=0.0
    # Else: run 2 simulations, compute delta
```

**Disclaimer stored in `reasoning`**: "Cache-aware scheduling routes requests to instances with matching prefix cached. Actual improvement depends on replica count and cache eviction rate."

---

## Schema Changes

### `TrafficProfile` — `src/planner/shared/schemas/specification.py` (add one field)

`system_prompt_tokens` belongs on `TrafficProfile`, not `DeploymentIntent`. `TrafficProfile` is generated after intent extraction (once `use_case` is known), making it the right place for use-case-derived values.

```python
# In TrafficProfile (src/planner/shared/schemas/specification.py):
system_prompt_tokens: int = 0
```

**Population flow** — `TrafficProfileGenerator.generate_profile()` in `src/planner/specification/traffic_profile.py`:
```python
SYSTEM_PROMPT_TOKEN_DEFAULTS = {
    "chatbot_conversational": 400,
    "code_completion": 600,
    "code_generation_detailed": 600,
    "rag_qa": 200,
    "long_document_summarization": 0,
    "research_legal_analysis": 0,
    # others default to 0
}

# At the end of generate_profile(), after the profile is built:
profile.system_prompt_tokens = SYSTEM_PROMPT_TOKEN_DEFAULTS.get(intent.use_case, 0)
return profile
```

`traffic_profile.system_prompt_tokens` is then available inside `config_finder.plan_all_capacities(traffic_profile, ...)` without any other schema changes.

### `CacheAffinityRecommendation` — add to `src/planner/shared/schemas/recommendation.py`

```python
@dataclass
class CacheAffinityRecommendation:
    policy: Literal["cache_aware", "round_robin"]
    reasoning: str
    simulated_ttft_improvement_pct: float  # positive = cache_aware is better
```

Export from `src/planner/shared/schemas/__init__.py` alongside existing exports.

### `DeploymentRecommendation` — `src/planner/shared/schemas/recommendation.py` (add four fields)

```python
cache_affinity_recommendation: CacheAffinityRecommendation | None = None
# None when: system_prompt_tokens == 0, candidate is not top-3 balanced, or simulator_mode is True
# NOT gated on replica count. Cache affinity measures prefix KV cache hit rate within a single
# instance — meaningful for any deployment size. Multi-replica routing decisions are out of scope.

kv_allocation_failure_rate: float = 0.0   # failures / total_requests (from simulation)
preemption_rate: float = 0.0              # preemptions / total_requests (from simulation)

@property
def reliability_status(self) -> Literal["ok", "warning", "critical"]:
    if self.kv_allocation_failure_rate > 0.02:
        return "critical"    # >2% requests fail to get KV blocks
    if self.kv_allocation_failure_rate > 0 or self.preemption_rate > 0.05:
        return "warning"     # any KV failures or >5% preemptions
    return "ok"

latency_source: Literal["simulation", "database"] = "database"
# Renamed from 'source' to avoid collision with BenchmarkData.source (tool origin)
# Surfaces in UI as a pill badge: '🔬 sim' (blue) or '📊 db' (gray), always visible
```

**Threshold rationale**: 2% KV allocation failure rate is a hard threshold — failed allocations mean requests are dropped or degrade significantly. 5% preemption is a soft threshold — preemptions increase latency but don't drop requests. Both are conservative starting points to be tuned once we have real workload data from inference-sim runs.

**Integration with Scorer**: `SimulationResult.ttft_p95_ms`, `itl_p95_ms`, and `e2e_p95_ms` replace the DB-sourced `bench.ttft_p95`, `bench.itl_p95`, `bench.e2e_p95` as inputs to `Scorer.score_latency()`. All downstream scoring logic in `scorer.py` remains unchanged.

---

## New API Endpoint

`POST /api/v1/explore-config` in `src/planner/api/routes/explore.py`

```python
# Request: explicit config fields — UI sends these directly from session-state recommendation.
# No server-side recommendation storage needed.
class ExploreConfigRequest(BaseModel):
    model: str
    gpu_type: str
    tensor_parallelism: int
    gpu_count: int                # GPUs per replica (from original recommendation)
    prompt_tokens: int
    output_tokens: int
    qps: float                    # must be > 0; validated server-side
    replicas: int                 # must be >= 1
    prefix_tokens: int = 0

# Response (success)
class ExploreConfigResponse(BaseModel):
    simulation: SimulationResult
    monthly_cost_usd: float          # recalculated: gpu_count × replicas × hourly_rate × 730
    reliability_status: str
    cache_affinity_recommendation: CacheAffinityRecommendation | None

# Error responses:
#   400 Bad Request  — qps <= 0, replicas < 1, or unknown model/gpu combination
#   503 Unavailable  — simulation binary not found (message: "Simulation unavailable;
#                      recommendation card values remain valid")
#   500 Server Error — unexpected simulation failure
```

---

## UI Changes

### Recommendation Cards

Each card gains two new rows below the existing TTFT/ITL/E2E p95 block:

```
Cache Affinity   cache-aware → 28% faster TTFT p95  (simulated)
Reliability      ✅ No KV failures · 0.1% preemptions
```

Warning state:
```
Reliability      ⚠️ 8% preemptions at 9 QPS — consider adding 1 replica
```

Critical state:
```
Reliability      🔴 3% KV allocation failures — insufficient GPU memory for this load
```

A small pill badge next to TTFT p95 indicates the data source: `🔬 sim` (blue) for live simulation, `📊 db` (gray) for static benchmarks. Always visible (not hover-only) to maintain transparency about prediction confidence.

### Explore Panel (Streamlit expander per card)

Opened via an "Explore scenarios" expander below each card. Layout:

**Preset scenario buttons** (run immediately on click, no extra interaction):
- `[ Recommended ]` — base config (always shown as reference)
- `[ 2× Traffic ]` — qps×2, replicas+1
- `[ Peak Load ]` — qps×3, replicas+2

**Custom scenario** (shown after clicking `[ Custom → ]`):
- QPS slider: 25%–200% of recommended QPS
- Replicas stepper: 1–(recommended+4), min 1
- Prefix tokens slider: 0–2048 (defaults to `system_prompt_tokens`)

**Results table** (one row per run scenario, stored in Streamlit `st.session_state` keyed by recommendation ID; persists for the session, resets when a new recommendation request is made):

| Scenario | QPS | Replicas | TTFT p95 | Cost/mo | Reliability |
|---|---|---|---|---|---|
| Recommended | 9 | 1 | 145ms | $5,840 | ✅ |
| 2× Traffic | 18 | 2 | 132ms | $11,680 | ✅ |
| Peak Load | 27 | 3 | 128ms | $17,520 | ⚠️ 6% preempt |

**Inline recommendation** when a scenario enters warning/critical:
> "💡 Peak Load config has 6% preemptions. Adding 1 more replica (→4 total) reduces this to 0.8%."

This recommendation is generated client-side on demand: when the UI renders a warning/critical row, it automatically calls `POST /api/v1/explore-config` with `replicas+1` (non-blocking, shows a spinner). Max one auto-simulation per warning scenario. Max 10 rows in the table before oldest non-base rows are evicted.

---

## Performance Design

Simulation calls are parallelized to keep total recommendation latency under ~500ms.

```
Sequential (naive):  20 candidates × ~50ms = ~1000ms
                     + 6 routing sims × ~50ms = ~1300ms total

Parallel (design):   20 candidates in ThreadPoolExecutor(max_workers=8) ≈ ~150ms
                     + 6 routing sims in ThreadPoolExecutor(max_workers=6) ≈ ~100ms
                     + scoring/ranking ≈ ~20ms
                     Total: ~270ms
```

Cache hits reduce this further for repeated configurations (common in the Explore panel).

---

## Inference-Sim Upstream Change

One small PR to `inference-sim`: add `--output json` flag.

`sim/metrics.go` gets a `PrintJSON()` method alongside the existing `Print()`. The CLI checks `--output` flag and calls the appropriate method. Output schema mirrors `SimulationResult` above.

This is the only change to inference-sim. All other logic lives in the planner.

---

## Implementation Order

1. **inference-sim upstream**: Add `--output json` flag (`sim/metrics.go` → `PrintJSON()` method). Run manually with a sample invocation to validate JSON output matches the schema contract above before writing any Python.
2. **Makefile**: Add `setup-inference-sim` target; call it (non-fatal) from `setup-backend`.
3. `simulation/client.py` — subprocess wrapper, JSON parsing, QPS≤0 guard (return None), 10s timeout, fallback returns None.
4. `simulation/cache.py` — module-level LRU, max 2048 entries.
5. `simulation/router.py` — two-simulation cache affinity comparison.
6. **Schema**: add `system_prompt_tokens: int = 0` to `TrafficProfile` (`src/planner/shared/schemas/specification.py`).
7. **Schema**: add `CacheAffinityRecommendation` and new fields on `DeploymentRecommendation` (`src/planner/shared/schemas/recommendation.py`); export from `__init__.py`.
8. **`TrafficProfileGenerator.generate_profile()`** (`src/planner/specification/traffic_profile.py`): set `profile.system_prompt_tokens = SYSTEM_PROMPT_TOKEN_DEFAULTS.get(intent.use_case, 0)` before returning the profile. No other files need to change to thread this value through.
9. **`config_finder.py`** (`plan_all_capacities()`): wrap existing DB candidate query with parallel `SimulationClient.simulate()` calls; use simulation result for latency if non-None, else use DB latency; set `latency_source` accordingly. After ranked lists, run cache affinity simulation for top-3 balanced.
10. **`POST /api/v1/explore-config`** in `src/planner/api/routes/explore.py` — register in `routes/__init__.py` and `app.py`.
11. **UI**: reliability status row, cache affinity row, `🔬 sim` / `📊 db` badge on recommendation cards.
12. **UI**: Explore panel — preset scenario buttons, results table in `st.session_state`, auto-simulate warning rows.

---

## Terminology

Throughout this document, **"cache affinity simulation"** refers to the two-run comparison (`prefix_tokens=0` vs `prefix_tokens=N`) used to estimate the benefit of routing requests to instances with a warm prefix cache. This is distinct from network-level load balancing (routing policy), which cannot be simulated with the single-instance inference-sim binary.

---

## Open Questions

- Should `system_prompt_tokens` be surfaced to users during intent extraction ("Do you use a system prompt? Roughly how long?") or always derived from use-case defaults? (Prototype uses defaults only.)
- Should reliability thresholds (2% KV failure, 5% preemption) be use-case specific — e.g., stricter for high-latency-priority deployments?
- Top-3 balanced candidates receive cache affinity simulation. If user is viewing the "lowest cost" tab, should we also run it for the top-3 lowest-cost candidates?
