# Comparative Analysis: LLM GPU/Memory Estimation Tools

## Tools Under Comparison

| Tool | Type | Approach |
|------|------|----------|
| **vllm_config_estimator** | Flask/CLI app | Physics-based + roofline analysis → vLLM config generator |
| **gpu-calc** | Single-file HTML app | Interactive calculator → GPU sizing dashboard |
| **llm-d-planner** | Full-stack system | Benchmark-driven + AI intent extraction → Kubernetes deployment |

---

## 1. Core KV Cache Formula

All three tools converge on the same fundamental formula for standard attention (MHA/GQA):

```
per_token_bytes = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

The `2×` factor accounts for separate K and V tensors. TP shards the KV heads:

| Tool | TP Sharding | Source |
|------|-------------|--------|
| **vllm_config_estimator** | `kv_heads_per_shard = max(1, num_kv_heads // tp)` | `vllm_start_config_from_estimate.py:745` |
| **gpu-calc** | Applied at replica-level (`vram × tp` in budget) | `index.html:2540` |
| **llm-d-planner** | Enumerated over valid TP divisors of `num_attention_heads` | `capacity_planner.py:795-813` |

**Key difference:** vllm_config_estimator and llm-d-planner explicitly shard KV heads by TP in the per-sequence calculation. gpu-calc applies TP by multiplying total VRAM and computing KV budget from the pooled resource — mathematically equivalent but loses per-head granularity.

### Advanced Architectures (MLA, Hybrid)

| Architecture | gpu-calc | vllm_config_estimator | llm-d-planner |
|-------------|----------|----------------------|---------------|
| **GQA/MQA** | Exact formula with `numKvHeads` | Exact, reads from HF config | Exact, reads `num_key_value_heads` |
| **MLA (DeepSeek V2/V3)** | `(kvLoraRank + ropeHeadDim) × kvPrec × layers` | Not explicitly handled | `(kv_lora_rank + qk_rope_head_dim) × precision × layers` |
| **Hybrid sliding window (Gemma 3)** | Split: global layers × full ctx + sliding layers × min(ctx, window) | Not handled | Not handled |
| **Hybrid SSM (Gemma 4)** | `kvLayers` = attention-only layers | Not handled | Not handled |

gpu-calc handles the most diverse set of attention architectures, driven by its UI focus on comparing diverse models.

---

## 2. Weight Memory Formula

```
weight_memory = parameter_count × bytes_per_parameter × overhead_factor
```

| Precision | gpu-calc | vllm_config_estimator | llm-d-planner |
|-----------|----------|----------------------|---------------|
| FP16/BF16 | 2.0 B/param | 2.0 B/param | From safetensors |
| FP8/INT8 | 1.0 B/param | 1.0 B/param | From safetensors |
| INT4/FP4 | 0.5 B/param | 0.5 B/param | From safetensors |

**Overhead factors:**

| Tool | Overhead | Rationale |
|------|----------|-----------|
| **vllm_config_estimator** | **1.15× (15%)** | Framework buffers, alignment, loader overhead |
| **gpu-calc** | **None (1.0×)** | Accounted for in separate 8% overhead term |
| **llm-d-planner** | **None** | Uses exact safetensors file sizes; separate activation/non-torch constants |

**Key difference:** vllm_config_estimator folds all overhead into a single weight multiplier. gpu-calc separates weights and a flat overhead term (`max(0.5GB, vram × 0.08)`). llm-d-planner is the most precise — it reads exact byte counts from HuggingFace safetensor metadata and uses empirically validated per-architecture activation constants.

---

## 3. Total GPU Memory Budget

```
available_kv = total_vram × utilization - weight_memory - overhead
max_concurrent_requests = floor(available_kv / kv_per_request)
kv_per_request = per_token_bytes × context_length
```

| Term | gpu-calc | vllm_config_estimator | llm-d-planner |
|------|----------|----------------------|---------------|
| `utilization` | 0.95 (vLLM paging) / 0.85 (naive) | **0.90** (fixed safe threshold) | **0.90** (GPU util param) |
| `overhead` | `max(0.5, vram × 0.08)` per GPU | 0 (folded into 1.15× weight factor) | 5.5 GiB activation (dense) + 0.15–0.6 GiB non-torch |
| `kv_precision` | FP16 or FP8 (user selectable) | Inherits from model dtype | **FP8 default** (1 byte) |

**Utilization semantics differ:**
- gpu-calc's 0.95 applies *after* weight subtraction, representing PagedAttention's block-level fragmentation efficiency.
- vllm_config_estimator's 0.90 is a conservative ceiling on *total* VRAM usage.
- llm-d-planner's 0.90 is passed to vLLM's `gpu_memory_utilization` parameter (vLLM internally manages KV block fragmentation via PagedAttention).

These are not directly comparable — applying them to the same hardware will produce meaningfully different KV cache budget estimates.

---

## 4. Throughput and Latency Estimation

| Tool | Method | What it produces |
|------|--------|-----------------|
| **vllm_config_estimator** | **Roofline analysis** (arithmetic intensity vs hardware bounds) | TTFT (ms), ITL (ms), output TPS — theoretical bounds |
| **gpu-calc** | **Heuristic TPP index** (memory bandwidth proxy) + user-calibrated values | Relative GPU throughput ranking; optional calibrated TPS from user input |
| **llm-d-planner** | **Empirical benchmark lookup** (BLIS simulator data) + roofline fallback | Actual p90/p95/p99 TTFT/ITL/E2E latency from measured benchmarks |

### Roofline Model (shared by vllm_config_estimator and llm-d-planner)

Both use [BentoML's llm-optimizer](https://github.com/bentoml/llm-optimizer) for roofline analysis. The roofline determines whether a workload is **compute-bound** or **memory-bandwidth-bound**:

```
arithmetic_intensity = FLOPs / bytes_accessed

if arithmetic_intensity < (TFLOPS / bandwidth):
    # Memory-bound (decode phase, common case)
    time = bytes_accessed / bandwidth
else:
    # Compute-bound (prefill phase with large batch)
    time = FLOPs / (TFLOPS × MFU)
```

**Prefill FLOPs** (per layer, per sequence):
```
prefill_flops = 2 × seq_len × (d_model² + 2 × d_model × d_kv) × num_layers   # QKV projections
              + 2 × num_heads × seq_len² × d_head × num_layers                 # attention scores
              + 2 × seq_len × 2 × d_model × d_ff × num_layers                  # MLP
```

**Decode FLOPs** (per token, typically memory-bound):
```
decode_flops = single_token_forward_pass_flops + attention_over_kv_context_flops
```

**MFU constants** (vllm_config_estimator `performance.py:301-302`):
- Prefill MFU: **0.45**
- Decode MFU: **0.30**

llm-d-planner prefers real benchmark data (BLIS) over roofline, falling back to roofline only for models not in the database.

---

## 5. Tensor Parallelism Selection

| Tool | TP values | Selection logic |
|------|-----------|-----------------|
| **vllm_config_estimator** | Any divisor of `num_gpus` (per-node constraint enforced) | Minimizes TP to fit weights; prefers maximizing DP for throughput, TP for latency |
| **gpu-calc** | Powers of 2: {1, 2, 4, 8} | Smallest TP where `weights/tp + overhead < vram × 0.95` |
| **llm-d-planner** | Divisors of `num_attention_heads` | Memory-feasibility gate: first TP where `allocatable_kv_cache > 0` |

**Architectural insight:** llm-d-planner's constraint is the most architecturally correct — TP must evenly divide attention heads to avoid head imbalance across devices. vllm_config_estimator enforces that TP spans only within a single node. gpu-calc's power-of-2 constraint is practical but doesn't reflect the attention head divisibility requirement.

---

## 6. Scoring and Ranking

| Tool | Ranking approach |
|------|-----------------|
| **vllm_config_estimator** | Ranks parallelism topologies by {latency, balanced, throughput} using TP/PP/DP sort order; no GPU ranking |
| **gpu-calc** | No automatic ranking; GPU Explorer bubble chart with user-selectable axes (`$/TPP`, bandwidth, VRAM); warning system |
| **llm-d-planner** | **4-dimension multi-criteria scoring** → 5 ranked recommendation lists |

### llm-d-planner Scoring (most sophisticated)

```python
balanced_score = (accuracy × 0.40 + price × 0.40 + latency × 0.10 + complexity × 0.10)
              × scalability_factor
```

**Price score** (non-linear, exponent 0.7):
```
price_score = 100 × (1 - (cost / max_cost)^0.7)
```

**Latency score** (capped range scoring with penalty zone):
```
if actual ≤ min:      score = 100
elif actual ≤ max:    score = 100 - (actual - min) / (max - min) × 40   # linear 100→60
else:                 score = max(0, 60 - (actual / max - 1) × 60)       # penalty zone

composite_latency = TTFT × 0.34 + ITL × 0.33 + E2E × 0.33
```

**Scalability penalty** (unique to llm-d-planner):
```
replicas:  1 → 100%,  2-3 → 98%,  4-6 → 95%,  7-10 → 90%,  11-20 → 80%,  21+ → 65%
```

gpu-calc's GPU Explorer computes `usdPerTpp = price / tpp` (cost per throughput-planning-index) as a cost-efficiency proxy, but TPP is a relative heuristic, not a calibrated metric.

---

## 7. Input Parameters

| Parameter | gpu-calc | vllm_config_estimator | llm-d-planner |
|-----------|----------|----------------------|---------------|
| Model size (B params) | User input / preset | Auto from HuggingFace | Auto from HuggingFace |
| Architecture (layers, heads, dim) | User input / HF fetch | Auto from HF config.json | Auto from HF AutoConfig |
| Context length | User input (K tokens) | `--input-len` + `--output-len` | Via SLO template (use case) |
| Batch / concurrent users | Concurrent users | Not direct (inferred from throughput) | User count → QPS formula |
| GPU type + VRAM | Dropdown (30+ GPUs) | `--gpu` flag (11 GPUs) | PostgreSQL benchmark coverage |
| Precision | FP16/FP8/INT4 | `--dtype` + `--quantization` | From benchmark data |
| SLO targets | Not specified | `--constraints ttft:p95<2s;itl:p95<50ms` | Derived from use case template |
| Cost parameters | Cloud rate, on-prem cost, electricity | Not included | GPU cost per hour (catalog) |

**Input abstraction level differs significantly:**
- gpu-calc: raw hardware parameters
- vllm_config_estimator: model + hardware + workload + SLO constraints
- llm-d-planner: business intent (use case, user count, latency priority) → everything derived

---

## 8. Output

| Output | gpu-calc | vllm_config_estimator | llm-d-planner |
|--------|----------|----------------------|---------------|
| VRAM breakdown | Yes (weights, KV, overhead) | Implicit in validation | No (internal only) |
| GPU count | Yes | Yes (via TP/PP/DP) | Yes (TP × replicas) |
| vLLM CLI args | No | **Yes (primary output)** | Partially (in YAML) |
| KServe/vLLM YAML | No | No | **Yes (primary output)** |
| Cost estimates | **Yes (5yr cloud vs on-prem)** | No | Yes (monthly USD) |
| Performance estimates | Heuristic TPP | TTFT/ITL/TPS (roofline) | TTFT/ITL/E2E p95 (benchmarks) |
| Ranked recommendations | No | 3 profiles (latency/balanced/throughput) | **5 ranked lists** |
| Deployment-ready | No | No | **Yes (one-click KServe deploy)** |

---

## 9. Key Constants Comparison

| Constant | gpu-calc | vllm_config_estimator | llm-d-planner |
|----------|----------|----------------------|---------------|
| GPU utilization ceiling | 0.95 (PagedAttention), 0.85 (naive) | **0.90** | **0.90** |
| Weight overhead | None (1.0×) | **1.15×** | None (exact safetensors) |
| Activation memory | Not modeled | 1.20× multiplier over weights | **5.5 GiB empirical** (dense) |
| Non-torch overhead | `max(0.5, vram×0.08)` | 0 (folded into weight overhead) | **0.15–0.6 GiB explicit** |
| Replica headroom | Not modeled | Not modeled | **1.2× (20%)** |
| Default KV precision | FP16 (switchable to FP8) | Inherits from model dtype | **FP8 default** |
| Power-on hours/month | 730 | N/A | 730 |

---

## 10. Similarities

1. **KV cache formula is universal.** All three implement the same `2 × layers × kv_heads × head_dim × tokens × bytes` formula. Differences are implementation details.

2. **GQA awareness.** All three correctly use `num_key_value_heads` (not `num_attention_heads`) for KV cache sizing — critical for models like Llama 3 (8 KV heads vs 32 query heads, a 4× difference).

3. **TP sharding of KV.** All three distribute KV cache across TP ranks rather than replicate it.

4. **Power-of-2 TP preference.** All three favor TP ∈ {1, 2, 4, 8} in practice, matching NVLink/PCIe topology and NCCL ring all-reduce efficiency.

5. **HuggingFace as the architecture source.** vllm_config_estimator and llm-d-planner both pull exact architecture parameters from HF Hub. gpu-calc supports HF fetch but also accepts manual input.

6. **Roofline for theoretical bounds.** Both vllm_config_estimator and llm-d-planner use the same underlying roofline model (BentoML llm-optimizer) for compute/memory bound analysis.

7. **p95 as the SLO percentile.** Both vllm_config_estimator and llm-d-planner use p95, consistent with GuideLLM conventions.

---

## 11. Key Differences

| Dimension | gpu-calc | vllm_config_estimator | llm-d-planner |
|-----------|----------|----------------------|---------------|
| **Memory accuracy** | Formula-based (good) | Formula + 1.15× overhead (conservative) | Empirically validated constants (most accurate) |
| **Latency source** | Heuristic bandwidth proxy | Roofline (theoretical) | Real benchmarks (most accurate) |
| **Architecture coverage** | Widest (MLA, hybrid SSM, hybrid sliding, MoE) | Standard + GQA | Standard + MLA |
| **User abstraction** | Technical (raw parameters) | Semi-technical (model + hardware) | Business intent (use case) |
| **Deployment integration** | None | vLLM CLI args only | End-to-end KServe deployment |
| **Cost modeling** | Detailed (cloud vs on-prem, 5yr, electricity) | None | Basic (monthly cloud cost) |
| **Activation memory** | **Not modeled** | Weight × 1.20 multiplier (approximate) | Per-architecture empirical constants (most accurate) |
| **Throughput planning** | Memory bandwidth proxy (TPP) | Roofline TPS (theoretical ceiling) | Benchmark-driven QPS + 20% headroom |
| **Recommendation output** | None (user-driven exploration) | 3 vLLM config profiles | 5 multi-criteria ranked lists |

---

## 12. Design Philosophy

**gpu-calc** is a **first-principles calculator**: it exposes the raw math to the user, supports the widest range of architectures, and emphasizes hardware comparison and cost trade-offs. Best for teams evaluating GPU procurement decisions or validating back-of-envelope calculations.

**vllm_config_estimator** is a **configuration advisor**: it takes a specific model + hardware pair and generates production-ready vLLM CLI arguments with validation. Best for engineers who have already decided on hardware and need operationally correct serving parameters.

**llm-d-planner** is a **deployment system**: it abstracts hardware entirely from the user, translates business requirements to SLO targets, queries real benchmark data, and generates Kubernetes manifests. Best for platform teams deploying LLMs at scale where the path from requirements to running cluster needs to be automated.

The three tools occupy distinct positions on the **abstraction-vs-accuracy** spectrum: gpu-calc maximizes user control at the cost of requiring hardware expertise; llm-d-planner maximizes automation at the cost of being constrained to its benchmark database; vllm_config_estimator sits in between, providing principled defaults with expert override paths.

---

## 13. Convergence: Making llm-d-planner the Canonical Backend

The goal: gpu-calc and vllm_config_estimator should be able to delegate all memory estimation, KV cache calculation, and performance estimation to llm-d-planner as a single source of truth, retaining only their own UI layers.

### 13.1 Capability Gaps in llm-d-planner Today

| Capability | gpu-calc | vllm_config_estimator | llm-d-planner |
|---|---|---|---|
| Hybrid sliding window KV (Gemma 3, Mistral) | ✅ | ❌ | ❌ |
| Hybrid SSM KV (Gemma 4) | ✅ | ❌ | ❌ |
| MLA KV (DeepSeek V2/V3) | ✅ | ❌ | ✅ |
| VRAM breakdown as API output | ✅ | ✅ (validation msgs) | ❌ internal only |
| vLLM serving parameters (max_num_seqs, batch tokens) | ❌ | ✅ primary output | ❌ |
| Pipeline parallelism (TP×PP×DP) | ❌ | ✅ | ❌ TP only |
| Multi-node topology validation | ❌ | ✅ hard error | ❌ |
| On-prem TCO / multi-year cost model | ✅ | ❌ | ❌ |
| Roofline as first-class estimate | ❌ heuristic | ✅ | fallback only |
| Quantization auto-detection from HF | ❌ | ✅ | ❌ benchmark-only |
| Per-token KV bytes as exposed metric | ✅ | partial | ❌ |

### 13.2 Points of Convergence

The three tools share these primitives and currently compute them independently. Each should be computed once, in llm-d-planner:

**Model architecture normalization.** All three independently call HuggingFace for `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `head_dim`. This should be a single endpoint returning a normalized struct that also classifies attention type (MHA/GQA/MQA/MLA/hybrid-sliding/hybrid-SSM).

**KV cache bytes per token.** The formula `2 × layers × kv_heads × head_dim × bytes_per_element` (with MLA and hybrid variants) is duplicated across all three. It should live in one canonical function exposed via API.

**VRAM budget decomposition.** All three decompose VRAM as:
```
total_vram = weights + activation + non_torch + kv_pool
```
with different constants and none expose the breakdown identically. A shared response format with each component named is the foundation for gpu-calc's summary cards and llm-d-planner's feasibility gating.

**Roofline performance model.** Both vllm_config_estimator and llm-d-planner already use BentoML's llm-optimizer. llm-d-planner calls it as a fallback; vllm_config_estimator calls it as the primary path. They should share one service call with a common request/response schema.

---

## 14. Action Items

### P0 — New API endpoints (unblocks the other two tools)

**A. `GET /api/v1/model-architecture/{hf_model_id}`**

Return normalized model architecture with attention classification. Replaces gpu-calc's manual model table and vllm_config_estimator's HF fetching:

```json
{
  "model_id": "deepseek-ai/DeepSeek-V3",
  "attention_type": "mla",
  "num_layers": 61,
  "num_attention_heads": 128,
  "num_kv_heads": 128,
  "head_dim": 128,
  "kv_lora_rank": 512,
  "qk_rope_head_dim": 64,
  "hidden_size": 7168,
  "intermediate_size": 18432,
  "num_experts": 256,
  "num_active_experts": 8,
  "quantization": "fp8"
}
```

**B. `POST /api/v1/memory-estimate`**

Expose what `capacity_planner.py` already computes internally. This is what both gpu-calc and vllm_config_estimator need from a backend — gpu-calc computes it in browser JS today, vllm_config_estimator in Flask:

```json
{
  "request": {
    "model_id": "meta-llama/Llama-3.1-70B",
    "gpu_type": "H100", "tensor_parallel": 2,
    "context_length": 8192, "kv_precision": "fp8"
  },
  "response": {
    "weight_memory_gib": 70.0,
    "kv_bytes_per_token": 131072,
    "kv_per_request_gib": 1.049,
    "activation_memory_gib": 5.5,
    "non_torch_memory_gib": 0.6,
    "total_overhead_gib": 6.1,
    "available_kv_pool_gib": 53.9,
    "max_concurrent_requests": 51,
    "fits": true
  }
}
```

**C. `POST /api/v1/performance-estimate`**

Elevate the existing roofline fallback path to a first-class endpoint. Returns benchmark data when available, roofline estimate otherwise:

```json
{
  "request": {
    "model_id": "...", "gpu_type": "H100",
    "tensor_parallel": 2,
    "prompt_tokens": 512, "output_tokens": 256
  },
  "response": {
    "ttft_p95_ms": 26.3,
    "itl_p95_ms": 10.7,
    "e2e_p95_ms": 3856,
    "output_tps": 247,
    "source": "benchmark",
    "confidence": "benchmarked"
  }
}
```

`source` is `"benchmark"` (BLIS DB hit) or `"roofline"` (llm-optimizer estimate). vllm_config_estimator would consume this instead of running llm-optimizer locally.

---

### P1 — Extend capacity_planner.py for missing architectures

**D. Hybrid sliding window KV (Gemma 3, Mistral)**

Currently absent from llm-d-planner. The formula from gpu-calc (`index.html:2500-2512`):

```python
if arch_type == "hybrid_sliding":
    global_bytes = global_layers * ctx * 2 * global_kv_heads * head_dim * bytes
    sliding_bytes = sliding_layers * min(ctx, window) * 2 * kv_heads * head_dim * bytes
    kv_bytes_per_token = (global_bytes + sliding_bytes) / ctx
```

Global attention layers store KV for the full context; sliding window layers store KV only for `min(ctx, window)` tokens. For Gemma 3 27B: 46 total layers, some global, some sliding with window=1024.

**E. Hybrid SSM KV (Gemma 4)**

For architectures where only a subset of layers have KV cache (attention layers), SSM layers have none. Add `kv_active_layers` to the architecture normalization and use it in place of `num_hidden_layers` when computing per-token KV bytes. For Gemma 4 26B: 46 total layers, 6 KV-active.

**F. Pipeline parallelism (multi-node deployments)**

llm-d-planner models only TP; vllm_config_estimator's most important correctness insight is that for multi-node deployments PP is the required axis (cross-node TP is unsupported in standard vLLM). Add PP to `GPUConfig` and to the feasibility check:

```
per_gpu_weight_memory = total_weights / (tp × pp)
```

With a hard validation error if TP > gpus_per_node.

---

### P2 — vLLM serving parameter generation

**G. `POST /api/v1/vllm-config`**

vllm_config_estimator's primary output — tuned vLLM CLI args — is genuinely useful and absent from llm-d-planner. The sizing logic from `vllm_start_config_from_estimate.py:761-881` maps cleanly to a new endpoint:

```json
{
  "tensor_parallel_size": 2,
  "pipeline_parallel_size": 1,
  "gpu_memory_utilization": 0.90,
  "max_num_seqs": 48,
  "max_num_batched_tokens": 4096,
  "kv_cache_dtype": "fp8",
  "enable_chunked_prefill": true,
  "max_model_len": 8192
}
```

`max_num_seqs` and `max_num_batched_tokens` are sized by GPU VRAM tier (vllm_config_estimator uses 64/4096 for 80GB+, 48/2048 for 48GB, 24/1024 for smaller). This would also improve the KServe YAML that llm-d-planner currently generates with default vLLM parameters.

---

### P3 — Cost model

**H. On-prem TCO in recommendation response**

Add on-prem cost fields alongside the existing `cost_per_month_usd`:

```json
{
  "cost_cloud_monthly_usd": 4380,
  "cost_onprem_monthly_amortized_usd": 833,
  "cost_onprem_power_monthly_usd": 127,
  "cloud_breakeven_months": 34
}
```

Formula from gpu-calc (`index.html:2609-2611`):
```
onprem_monthly = (gpu_price × count / amort_months) + (count × 0.7kW × 730hr × $/kWh)
breakeven_months = ceil(gpu_price × count / (cloud_monthly - onprem_monthly))
```

---

## 15. Contribution Map

| Tool | Contributes to llm-d-planner backend | Retains as UI layer |
|---|---|---|
| **gpu-calc** | Hybrid architecture KV formulas (D, E); on-prem cost model (H); GPU specs for 30+ hardware types | Hardware comparison dashboard, GPU Explorer bubble chart, cloud vs on-prem cost comparison |
| **vllm_config_estimator** | TP×PP topology logic (F); vLLM serving parameter tuning (G); quantization auto-detection | Thin wrapper calling `/api/v1/vllm-config` instead of running its own math |
| **llm-d-planner** | Canonical backend: empirical benchmark DB, memory estimation service, roofline service, multi-criteria scoring, YAML generation | Keeps chat UI and Kubernetes deployment flow |

The cleanest convergence path: implement A–C first (three new endpoints), giving the other two tools a concrete backend to call. D–G fill correctness gaps. H is additive value with low risk.
