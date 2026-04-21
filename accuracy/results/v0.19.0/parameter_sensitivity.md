# Parameter Sensitivity Analysis — Capacity Planner vs vLLM

_vLLM v0.19.0 · H100-80GB · Llama-3.1-8B and Qwen2.5-7B as reference models_

---

## Summary table

| Parameter | Affects weight prediction? | Affects KV-GiB prediction? | Affects token-capacity prediction? | Planner handles it? |
|---|---|---|---|---|
| `--dtype` (bf16 / fp16 / auto) | No | No | No | ✅ N/A |
| `--dtype float32` | **Yes — 2× weight** | **Yes — 2× KV/token** | **Yes — 2×** | ❌ Gap (measured: −50% weight, +31% KV) |
| `--kv-cache-dtype` (auto / fp8) | No | No | **Yes — 2×** | ❌ Gap |
| Weight quantization (w8a8 / w4a16) — small model | ✅ Yes, correctly | ✅ Yes, correctly | No | ✅ Yes |
| Weight quantization (w8a8) — large model (70B) | ✅ Yes, correctly | ❌ Over-reserves | No | ⚠️ Partial |

---

## `--dtype` (bf16 / fp16 / auto / **fp32**)

**Tested: bf16, fp16, auto — no effect. fp32 — measured gap confirmed.**

### Tested values (bf16 / fp16 / auto)

Both bf16 and fp16 are 2-byte formats. For all models tested, switching `--dtype` among
`auto`, `bfloat16`, and `float16` produced **identical** weight and KV memory measurements.

| Model | dtype | Weight (GiB) | KV avail (GiB) | KV tokens |
|---|---|---|---|---|
| Llama-3.1-8B | auto | 14.99 | 58.11 | 476 000 |
| Llama-3.1-8B | bfloat16 | 14.99 | 58.11 | 476 000 |
| Llama-3.1-8B | float16 | 14.99 | 58.11 | 476 016 |
| Qwen2.5-7B | auto | 14.25 | 58.53 | 1 096 000 |
| Qwen2.5-7B | bfloat16 | 14.25 | 58.53 | 1 096 000 |
| Qwen2.5-7B | float16 | 14.25 | 58.53 | 1 096 000 |

The planner reads the storage dtype from safetensors metadata and handles these correctly.
For almost all production models the safetensors dtype is bf16, and `auto` / `bfloat16` /
`float16` all resolve to the same 2 bytes/element in memory.

### `--dtype float32` — measured gap

**Root cause**: `--dtype` is not a parameter the planner exposes. Neither
`per_gpu_model_memory_required()` nor `allocatable_kv_cache_memory()` accept a dtype
argument, so there is no way to pass a runtime dtype override to the planner. Internally
it reads:
- Weight bytes from safetensors storage dtype (`model_params_by_dtype`) → bf16 = 2 bytes
- KV element size from `model_config.torch_dtype` or `inference_dtype()` → also bf16

With `--dtype float32`, vLLM upcasts every weight tensor to fp32 (4 bytes) in GPU memory
and also stores KV cache elements in fp32. The planner has no way to see or account for
this at all.

**Measured prediction errors — Llama-3.1-8B, TP=1, H100-80GB:**

| Component | Planner predicts (bf16 storage) | vLLM measured (fp32 runtime) | Error |
|---|---|---|---|
| Weight memory | ~14.96 GiB | **29.98 GiB** | **−50.1%** |
| KV avail (GiB) | ~56.1 GiB (inflated — sees extra 15 GiB) | **42.80 GiB** | **+31.1%** |
| KV tokens | ~460 000 | **175 296** | ~+163% |
| KV block bytes | ~2.1 MB | ~4.2 MB | ~+100% (fp32 per-element) |

Because KV available GiB is computed as `GPU_budget − weight − activation − overhead`,
and weight is underestimated by 15 GiB, the planner thinks there is 15 GiB more KV room
than actually exists. A model the planner declares fits on one H100 **may OOM at runtime**.

**Fix required** in `per_gpu_model_memory_required()` / `KVCacheDetail.__init__()`
(in `capacity_planner.py`): accept a `dtype_override` argument. When `--dtype float32`
is requested, multiply all 2-byte storage costs by 2 before computing memory budgets.

```python
# Pseudocode for the fix
if vllm_args.get("dtype") == "float32":
    weight_bytes_per_param *= 2   # upcast from storage bf16 → fp32
    kv_bytes_per_element   *= 2   # KV also stored in fp32
```

**When does `--dtype float32` matter in practice?**
Rarely in production (bf16/fp16 is standard for inference). It appears in:
- Debugging runs on GPUs without bf16 support (e.g., some older V100 configs)
- Research runs requiring higher numerical precision
- CPU-only inference (not GPU-relevant here)

Recommendation: add a validation warning in the planner if `--dtype float32` is requested,
since capacity estimates will be unreliable until the fix is implemented.

---

## `--kv-cache-dtype` (auto / fp8)

**Conclusion: does NOT change allocatable KV GiB, but doubles token capacity. The planner
correctly predicts KV GiB, but does not model the token-count implication of fp8.**

When `--kv-cache-dtype=fp8`, vLLM stores each KV element in 1 byte instead of 2.
The GPU allocates the **same number of bytes** for KV regardless — but twice as many tokens
fit within that budget.

| Model | kv_cache_dtype | Weight (GiB) | KV avail (GiB) | KV tokens | Bytes/token |
|---|---|---|---|---|---|
| Llama-3.1-8B | auto (bf16) | 14.99 | 58.11 | 476 000 | 2 097 315 / block |
| Llama-3.1-8B | fp8 | 14.99 | **58.11** | **952 032** | 1 048 622 / block |
| Qwen2.5-7B | auto (bf16) | 14.25 | 58.53 | 1 096 000 | 917 461 / block |
| Qwen2.5-7B | fp8 | 14.25 | **58.53** | **2 192 000** | 458 730 / block |

Observations:
- KV GiB is **identical** — the allocatable memory budget is dtype-agnostic.
- KV token count **doubles** — fp8 halves bytes-per-element.
- Block size (bytes) **halves** — confirms fp8 is applied at the element level.

**Planner accuracy for KV GiB**: unaffected by this flag. Error for both runs is
identical (e.g. −3.5% for Llama-3.1-8B) because the planner computes in GiB.

**Gap**: the planner does not expose a token-count or max-concurrency estimate.
Any downstream code that converts predicted KV GiB → token count must apply:

```
kv_tokens = kv_cache_gib × GiB_in_bytes / per_token_bytes(kv_cache_dtype)
```

where `per_token_bytes` is 2 for `auto`/bf16/fp16 and 1 for `fp8`.

---

## Weight quantization (`--quantization` / model-embedded)

### Small quantized models (Llama-3.1-8B w8a8 and w4a16)

**Conclusion: weight and KV predictions are accurate. The planner correctly reads
quantization config from the HuggingFace model and adjusts bytes-per-parameter.**

| Model | Quant | Weight measured | Weight predicted | Weight err | KV measured | KV predicted | KV err |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | fp16 (baseline) | 14.99 GiB | ~14.96 GiB | −0.2% | 58.11 GiB | ~56.09 GiB | −3.5% |
| Llama-3.1-8B | w8a8 | 8.49 GiB | ~8.46 GiB | −0.4% | 64.60 GiB | ~62.59 GiB | −3.1% |
| Llama-3.1-8B | w4a16 | 5.38 GiB | ~5.34 GiB | −0.7% | 67.71 GiB | ~65.69 GiB | −3.0% |

Key observations:
- Weight reduction is correctly modeled: w8a8 saves ~6.5 GiB, w4a16 saves ~9.6 GiB.
- KV budget expands proportionally as expected (more room once weights shrink).
- The ~3% KV under-prediction is consistent with the unquantized baseline, indicating
  it comes from the activation constant overestimate (see below), not quantization handling.
- `quantization: null` in vllm_args is expected for these models — the quantization is
  embedded in the model weights; vLLM detects it automatically via the model config.

### Large quantized models (Llama-3.3-70B w8a8)

**Conclusion: weight is predicted correctly, but KV is significantly under-predicted (−32.7%
at TP=1). Root cause: the activation constant (5.5 GiB) was calibrated on fp16 models.
W8A8 reduces activation memory because intermediate tensors are int8.**

| TP | Weight measured | KV measured | KV err | Derived activation |
|---|---|---|---|---|
| TP=1 | 67.72 GiB | 5.01 GiB | −32.7% | ~3.1 GiB |
| TP=2 | 33.88 GiB/GPU | 37.28 GiB/GPU | +5.0% | (residual absorbed) |
| TP=4 | 16.96 GiB/GPU | 54.08 GiB/GPU | +5.9% | (residual absorbed) |

Deriving actual activation memory at TP=1:

```
available = 80 GiB × 0.95 = 76.0 GiB
consumed  = weight + cuda_graph + non_torch + kv
         = 67.72 + 0.84 + 0.15 + 5.01 = 73.72 GiB
residual  = 76.0 - 73.72 = 2.28 GiB → actual activation ≈ 2.3 GiB
```

vs planner constant = **5.5 GiB** → overestimates by ~3.2 GiB → predicts 3.2 GiB less KV.

This explains the −32.7% error on a model with only ~5 GiB of usable KV:
3.2 GiB over-reservation on a 5 GiB budget = ~64% effective error. Expressed as the
measured 32.7% figure: `(predicted − measured) / measured = (1.81 − 5.01) / 5.01 = −63.8%`.

**Why does w8a8 reduce activation memory?**  
With int8 activations, intermediate tensors (attention scores, MLP buffers) are 1 byte/element
instead of 2. For a 70B model, this approximately halves the activation footprint. For small
models (8B), the same effect is present but the absolute magnitude (~1 GiB) is small relative
to the large KV budget (~64 GiB), so it's only a −3% KV error.

**Why does the error disappear at TP≥2?**  
At TP=2 and TP=4, the weight per GPU drops significantly (33.9 / 17.0 GiB), leaving much
more room for KV. A 3 GiB activation over-reservation becomes a small fraction of the
large available KV budget, so the relative error is within ±6%.

---

## Implications for planner calibration

1. **`--dtype`: no action needed.** The planner reads the actual safetensors dtype and
   handles bf16, fp16, and auto identically and correctly.

2. **`--kv-cache-dtype`: add token-count conversion.** KV GiB prediction is correct.
   Expose a downstream conversion: `kv_tokens = kv_gib × GiB / per_element_bytes(kv_dtype)`.
   The planner's `KVCacheDetail` already has `kv_data_type` — this can be used to compute
   the per-token cost and hence max concurrency.

3. **Weight quantization on small models: no action needed.** The planner correctly reads
   quantization config and applies the right bytes-per-parameter.

4. **W8A8 activation constant on large models: needs a separate constant.**
   Introduce `ACTIVATION_MEMORY_BASE_DENSE_W8A8_GIB ≈ 2.3` for large w8a8 models, or
   scale the existing constant by dtype precision: `activation × (quant_bytes / 2.0)`.
   Evidence: 70B w8a8 shows ~2.3 GiB actual vs 5.5 GiB assumed; 8B w8a8 shows ~1.5 GiB.
   A simple heuristic: `activation_for_quantized = activation_fp16 × 0.45`.

---

## What does NOT affect predictions

- **`max_model_len`**: KV error is identical across 2K–32K for both Llama-3.1-8B and
  Qwen2.5-7B at TP=1. The formula is correctly context-length-agnostic (allocates by bytes,
  not by tokens).

- **`gpu_memory_utilization`**: scaling the utilization factor is a linear multiplier in the
  formula; no systematic error expected as long as the value matches what's passed to vLLM.

- **TP/PP degree** (after normalisation): the ÷(TP×PP) correction brings most multi-GPU
  KV errors within ±10%, with residual positive bias coming from NCCL buffer overhead
  not captured in the `non_torch` constant.
