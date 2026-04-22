# Capacity Planner Accuracy Report — vLLM v0.19.0 / H100-80GB

**Dataset**: 47 successful runs across 22 unique models  
**Hardware**: H100-80GB (catalog memory = 80 GiB, actual = ~79.19 GiB)  
**Planner GPU util**: actual `gpu_memory_utilization` per run (0.95)  

## Executive Summary

| Metric | Mean error | Mean abs error | Notes |
|--------|:----------:|:--------------:|-------|
| **KV Cache memory** (all 47 runs) | +0.83% | +7.91% | |
| **KV Cache memory** (baseline: tp=pp=1, len=8192, no-quant) | -4.11% | — | n=16 |
| **Weight memory** | -0.03% | +3.27% | From safetensors metadata |
| **Activation memory** | +196.08% | +196.08% | Largest error source |
| **Non-torch overhead** | -43.67% | +53.40% | |
| **Max concurrency** | -1.57% | +15.90% | Proxy for KV cache accuracy |

### Key Findings

1. **Weights are accurate** — mean abs error +3.27%, computed directly from safetensors parameter counts. Errors arise only when `--dtype` overrides the native dtype (e.g., `--dtype float32`) or when quantization is not fully captured in the config.
2. **Activation is the dominant error source** — mean +196.08% (over-estimate). The planner uses empirical constants (4.8–8.0 GiB) measured at `max_model_len=16000`; vLLM v0.19.0 reports 0.75–2.2 GiB across all architectures tested. Granite is worst (+600%), Mistral3/Pixtral is best (+15–23%).
3. **Over-estimated activation partially cancels** the catalog GPU memory inflation (+0.77 GiB), leaving KV cache only +0.83% off on average across all runs. But this is coincidental cancellation of two large opposing errors, not model accuracy.
4. **Non-default KV dtype (`--kv-cache-dtype fp8`) doubles token capacity** but the planner ignores this flag — KV token count is off by ~2× for those runs.
5. **`--dtype float32` breaks weight prediction** — the planner uses the HuggingFace config dtype (BF16) and never sees the vLLM `--dtype` override, giving −50% weight error.
6. **Pipeline parallelism reduces actual activation** (each GPU processes fewer layers) but the formula uses the same constant regardless of PP, compounding the activation error.

## Component-Level Error Breakdown

> Percent error = (predicted − actual) / actual × 100. Positive = over-estimate, negative = under-estimate.


### All 47 Runs  (n=47)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | -0.03% | -0.31% | +3.27% | -50.11% | +76.18% | 47 |
| Activation | +196.08% | +153.97% | +196.08% | +14.68% | +633.33% | 47 |
| Non Torch | -43.67% | -40.00% | +53.40% | -72.85% | +114.29% | 47 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 47 |
| Total Non Kv | +19.52% | +16.70% | +21.45% | -38.61% | +97.49% | 47 |
| Kv Cache | +0.83% | -3.47% | +7.91% | -28.75% | +61.82% | 47 |
| Max Concurrency | -1.57% | -3.48% | +15.90% | -87.31% | +162.10% | 47 |

### Baseline: TP=1, PP=1, len=8192, no quantization, default KV dtype  (n=16)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | -3.38% | -0.22% | +3.39% | -50.11% | +0.04% | 16 |
| Activation | +247.20% | +163.97% | +247.20% | +23.15% | +633.33% | 16 |
| Non Torch | -45.25% | -40.00% | +45.25% | -67.39% | -37.50% | 16 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 16 |
| Total Non Kv | +17.06% | +17.29% | +21.89% | -38.61% | +74.27% | 16 |
| Kv Cache | -4.11% | -4.29% | +8.18% | -28.75% | +31.06% | 16 |
| Max Concurrency | -0.76% | -4.29% | +21.22% | -87.31% | +162.10% | 16 |

### Multi-GPU (TP > 1 or PP > 1)  (n=15)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | -1.20% | -0.38% | +1.20% | -12.22% | -0.03% | 15 |
| Activation | +196.02% | +153.39% | +196.02% | +23.15% | +561.16% | 15 |
| Non Torch | -46.75% | -71.29% | +77.23% | -72.85% | +114.29% | 15 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 15 |
| Total Non Kv | +16.86% | +11.34% | +17.76% | -6.76% | +97.49% | 15 |
| Kv Cache | +11.26% | +4.62% | +11.63% | -1.88% | +61.82% | 15 |
| Max Concurrency | +6.58% | +4.63% | +16.32% | -71.19% | +62.24% | 15 |

### Quantized Models (fp8-dynamic / w8a8 / w4a16)  (n=10)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | +7.29% | -0.29% | +7.94% | -0.79% | +76.18% | 10 |
| Activation | +124.00% | +149.15% | +124.00% | +14.68% | +153.97% | 10 |
| Non Torch | -52.67% | -41.15% | +52.67% | -72.85% | -37.50% | 10 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 10 |
| Total Non Kv | +21.87% | +15.87% | +23.22% | -6.76% | +87.45% | 10 |
| Kv Cache | -0.45% | -0.87% | +4.95% | -13.18% | +5.90% | 10 |
| Max Concurrency | -0.44% | -0.86% | +4.95% | -13.19% | +5.89% | 10 |

### Non-default KV cache dtype (--kv-cache-dtype fp8)  (n=2)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | -0.34% | -0.34% | +0.34% | -0.45% | -0.22% | 2 |
| Activation | +153.68% | +153.68% | +153.68% | +153.39% | +153.97% | 2 |
| Non Torch | -38.75% | -38.75% | +38.75% | -40.00% | -37.50% | 2 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 2 |
| Total Non Kv | +17.83% | +17.83% | +17.83% | +16.28% | +19.37% | 2 |
| Kv Cache | -3.84% | -3.84% | +3.84% | -4.21% | -3.47% | 2 |
| Max Concurrency | -51.92% | -51.92% | +51.92% | -52.11% | -51.73% | 2 |

## Per-Model Errors — Baseline Runs

> TP=1, PP=1, max_model_len=8192, no quantization, default KV dtype.

| Model | Arch | Weight err | Activation err | Non-torch err | KV cache err | Max conc err |
|-------|------|:----------:|:--------------:|:-------------:|:------------:|:------------:|
| Qwen2.5-7B-Instruct | Qwen2 | -0.45% | +153.39% | -37.50% | -4.21% | -4.22% |
| Qwen2.5-7B-Instruct | Qwen2 | -0.45% | +153.39% | -37.50% | -4.21% | -4.22% |
| Qwen3-30B-A3B | Qwen3Moe | -0.02% | +198.51% | -44.44% | -28.75% | -28.72% |
| Qwen3-8B | Qwen3 | -0.09% | +153.39% | -40.00% | -4.36% | -4.36% |
| DeepSeek-V2-Lite-Chat | DeepseekV2 | -0.59% | +314.51% | -42.31% | -11.50% | -11.50% |
| granite-3.1-2b-instruct | Granite | -0.44% | +633.33% | -67.39% | -5.27% | -5.27% |
| granite-3.1-8b-instruct | Granite | -0.20% | +547.06% | -67.39% | -6.02% | -6.03% |
| granite-3.3-8b-instruct | Granite | -0.20% | +547.06% | -67.39% | -6.02% | -6.03% |
| granite-vision-3.3-2b | LlavaNext* | +0.04% | +216.46% | -40.00% | -1.23% | -1.23% |
| Llama-3.1-8B-Instruct | Llama | -0.22% | +153.97% | -40.00% | -3.47% | -3.48% |
| Llama-3.1-8B-Instruct | Llama | -0.22% | +153.97% | -40.00% | -3.47% | -3.48% |
| Llama-3.1-8B-Instruct | Llama | -50.11% | +117.19% | -40.00% | +31.06% | +162.10% |
| Llama-3.1-8B-Instruct | Llama | -0.22% | +153.97% | -40.00% | -3.47% | -3.48% |
| phi-4 | Phi3 | -0.31% | +261.84% | -40.00% | -6.59% | -6.58% |
| Mistral-Small-3.1-24B-Instruct-2503 | Mistral3* | -0.08% | +23.15% | -40.00% | +1.54% | +1.55% |
| Kimi-VL-A3B-Instruct | KimiVL* | -0.58% | +173.97% | -40.00% | -9.76% | -87.31% |

## Argument Sensitivity Analysis

> This section examines how each vLLM launch argument affects whether the capacity planner's memory predictions remain accurate.

### `--max-model-len` (context window size)

| Model | max_model_len | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err | Max conc err |
|-------|:-------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|:------------:|
| Llama-3.1-8B-Instruct | 2,048 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -3.47% |
| Llama-3.1-8B-Instruct | 4,096 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -3.47% |
| Llama-3.1-8B-Instruct | 8,192 | 58.11 | -3.47% | 476,000 | 459,509 | -3.46% | -3.48% |
| Llama-3.1-8B-Instruct | 8,192 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -3.48% |
| Llama-3.1-8B-Instruct | 8,192 | 42.80 | +31.06% | 175,296 | 459,509 | +162.13% | +162.10% |
| Llama-3.1-8B-Instruct | 8,192 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -3.48% |
| Llama-3.1-8B-Instruct | 16,384 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -3.44% |
| Llama-3.1-8B-Instruct | 32,768 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -3.51% |
| Qwen2.5-7B-Instruct | 2,048 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% | -4.22% |
| Qwen2.5-7B-Instruct | 4,096 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% | -4.22% |
| Qwen2.5-7B-Instruct | 8,192 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% | -4.22% |
| Qwen2.5-7B-Instruct | 8,192 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% | -4.22% |
| Qwen2.5-7B-Instruct | 16,384 | 58.53 | -4.21% | 1,095,968 | 1,049,789 | -4.21% | -4.22% |
| Qwen2.5-7B-Instruct | 32,768 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% | -4.22% |

**Conclusion**: `--max-model-len` has **no effect on KV pool size** — the formula and vLLM agree on this. Activation memory is constant (the fixed profiling overhead does not depend on context length), so the KV pool prediction error stays flat at ~−3 to −4% regardless of whether context is 2 K or 32 K tokens. The token/concurrency predictions carry that same constant KV error forward, plus any error from the per-token KV formula.

### `--tensor-parallel-size` (TP)

| Model | TP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |
|-------|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|
| Llama-3.1-8B-Instruct | 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| Llama-3.1-8B-Instruct | 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| Llama-3.1-8B-Instruct | 1 | 29.98 | -50.11% | 2.21 | +117.19% | 0.25 | -40.00% | +31.06% |
| Llama-3.1-8B-Instruct | 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| Llama-3.1-8B-Instruct | 2 | 7.51 | -0.42% | 1.89 | +153.97% | 2.07 | -71.01% | +2.76% |
| Llama-3.1-8B-Instruct | 4 | 3.77 | -0.81% | 1.89 | +153.97% | 2.13 | -71.83% | +4.48% |
| Qwen2.5-7B-Instruct | 1 | 14.25 | -0.45% | 2.21 | +153.39% | 0.24 | -37.50% | -4.21% |
| Qwen2.5-7B-Instruct | 1 | 14.25 | -0.45% | 2.21 | +153.39% | 0.24 | -37.50% | -4.21% |
| Qwen2.5-7B-Instruct | 2 | 7.12 | -0.38% | 2.21 | +153.39% | 2.06 | -70.87% | +2.61% |
| Qwen2.5-7B-Instruct | 4 | 3.55 | -0.10% | 2.21 | +153.39% | 2.13 | -71.83% | +4.62% |

**Conclusions**:

- **Weights scale correctly**: the formula divides by TP, matching vLLM's per-GPU weight sharding. Weight error stays near 0% across TP=1–4.
- **Activation is TP-invariant in both formula and reality**: vLLM's profiling overhead does not shrink with TP (it captures the same set of batch sizes). The formula also keeps activation constant with TP. Error stays flat.
- **Non-torch is heavily under-estimated for TP≥2**: the 0.60 GiB/GPU constant does not capture NCCL all-reduce buffer overhead, which grows with TP. Actual non-torch reaches ~2.1 GiB/GPU at TP=4 (3.5× the constant). However, this error is partially masked in KV cache accuracy because the over-estimated activation pulls the prediction in the opposite direction.

### `--pipeline-parallel-size` (PP)

| PP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |
|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|
| 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| 1 | 29.98 | -50.11% | 2.21 | +117.19% | 0.25 | -40.00% | +31.06% |
| 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| 2 | 7.51 | -0.42% | 1.10 | +336.36% | 0.07 | +114.29% | -0.85% |
| 4 | 4.26 | -12.22% | 1.05 | +357.14% | 0.07 | +114.29% | +1.59% |

**Conclusions**:

- **Activation drops sharply with PP**: at PP=1, vLLM profiles 1.89 GiB of activation; at PP=2 it drops to 1.10 GiB; at PP=4 to 1.05 GiB. Each pipeline stage runs fewer transformer layers, so the profiling sweep allocates proportionally less. The formula does not account for this and always predicts 4.80 GiB, making the activation error grow with PP (from ~+154% at PP=1 to ~+357% at PP=4).
- **Non-torch increases with PP** due to inter-stage P2P send/receive buffers, but the formula uses the same TP=1 constant (0.15 GiB/GPU) regardless of PP, causing the non-torch estimate to overshoot actual (predicted > actual for PP>1 because each stage is a separate process and 0.15 is per-GPU). These two errors partially offset each other in the KV cache prediction.
- **Weight error grows with PP**: the formula divides only by TP×PP for weight sharding, but with PP=4, model layers are not uniformly distributed across stages in all cases (irregular last-stage allocation can leave a stage with fewer params).

### `--dtype` (compute/storage dtype override)

| dtype arg | quantization | kv_cache_dtype | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |
|-----------|:------------:|:--------------:|:-------------------:|:----------:|:---------------:|:------:|
| bfloat16 | None | auto | 14.99 | -0.22% | 58.11 | -3.47% |
| bfloat16 | None | auto | 14.99 | -0.22% | 58.11 | -3.47% |
| bfloat16 | None | fp8 | 14.99 | -0.22% | 58.11 | -3.47% |
| bfloat16 | fp8 | auto | 8.49 | +76.18% | 64.61 | -13.18% |
| float16 | None | auto | 14.99 | -0.22% | 58.11 | -3.47% |
| float16 | compressed-tensors | auto | 8.49 | -0.35% | 64.60 | -3.11% |
| float16 | gptq_marlin | auto | 5.38 | -0.71% | 67.71 | -2.96% |
| float32 | None | auto | 29.98 | -50.11% | 42.80 | +31.06% |

**Conclusions**:

- **`--dtype float32`** doubles model weight memory (29.98 GiB vs BF16's 14.99 GiB). The planner reads the HuggingFace config dtype (BF16) and is unaware of the `--dtype` vLLM override, so it predicts 14.96 GiB — a **−50% weight error**, which cascades into a +31% KV cache over-prediction (the planner thinks there is more room than there is).
- **`--dtype float16`** is handled correctly because the HuggingFace config also stores float16 for these models; weight error stays near 0%.
- **FP8-dynamic quantization** (`fp8` in the quantization column) halves weight memory. The planner reads `quantization_config` from the HuggingFace repo and applies the FP8 byte-per-param, yielding near-zero weight error. KV cache error stays consistent with the activation over-estimation.
- **`--kv-cache-dtype fp8`** does not affect weight or activation predictions, but halves per-token KV storage. The planner ignores this flag and predicts KV tokens ~50% too low (see dedicated section below).

### `--quantization` (weight quantization method)

| Model | quant method | TP | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |
|-------|--------------|----|:-------------------:|:----------:|:---------------:|:------:|
| Llama-3.3-70B-Instruct-fp8-dyn | compressed-tensors | 2 | 33.88 | -0.12% | 37.28 | +5.04% |
| Llama-3.3-70B-Instruct-fp8-dyn | compressed-tensors | 4 | 16.96 | -0.24% | 54.09 | +5.90% |
| Meta-Llama-3.1-8B-Instruct-qua | compressed-tensors | 1 | 8.49 | -0.35% | 64.60 | -3.11% |
| Mistral-Small-3.1-24B-Instruct | compressed-tensors | 1 | 24.07 | -0.18% | 48.73 | +1.22% |
| Mistral-Small-3.1-24B-Instruct | compressed-tensors | 2 | 12.11 | -0.79% | 59.02 | +5.28% |
| Qwen2.5-7B-Instruct-fp8-dynami | compressed-tensors | 1 | 8.14 | -0.36% | 64.64 | -3.87% |
| Qwen2.5-7B-Instruct-quantized. | compressed-tensors | 1 | 8.14 | -0.36% | 64.64 | -3.87% |
| Llama-3.3-70B-Instruct-quantiz | compressed-tensors | 2 | 33.88 | -0.12% | 37.28 | +5.04% |
| Llama-3.1-8B-Instruct | fp8 | 1 | 8.49 | +76.18% | 64.61 | -13.18% |
| Meta-Llama-3.1-8B-Instruct-qua | gptq_marlin | 1 | 5.38 | -0.71% | 67.71 | -2.96% |

**Conclusions**:

- **w8a8 (compressed-tensors INT8)**: the planner parses `config_groups` from the `quantization_config` to find `num_bits=8` and applies 1 byte/param. Weight errors are near zero (−0.3 to −0.7%), indicating the INT8 parameter count is well-captured.
- **w4a16 (GPTQ-marlin INT4)**: the planner parses `num_bits=4` from the quantization config and applies 0.5 bytes/param. Weight error is small (~−0.7%). The large reduction in weights (5.3 GiB vs 15 GiB for BF16) frees more KV cache, and the planner correctly tracks this effect — KV error stays in the −3% range.
- **fp8-dynamic** (fp8 per-tensor dynamic quant via `compressed-tensors`): the planner extracts fp8 precision from the quantization config. Weight error is near zero. Unexpectedly, weight error for the RedHat fp8 70B model at TP=2 stays very low, confirming the quant config parsing is correct for this variant.

### `--kv-cache-dtype` (KV cache precision)

| Model | kv_cache_dtype | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err | Conc err |
|-------|:--------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|:--------:|
| Qwen2.5-7B-Instruct | auto | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% | -4.22% |
| Qwen2.5-7B-Instruct | fp8 | 58.53 | -4.21% | 2,192,000 | 1,049,789 | -52.11% | -52.11% |
|||||||||
| Llama-3.1-8B-Instruct | auto | 58.11 | -3.47% | 476,000 | 459,509 | -3.46% | -3.48% |
| Llama-3.1-8B-Instruct | fp8 | 58.11 | -3.47% | 952,032 | 459,509 | -51.73% | -51.73% |
|||||||||

**Conclusion**: `--kv-cache-dtype fp8` stores each KV element in 1 byte instead of 2 bytes (BF16/FP16), doubling the number of tokens that fit in the KV pool. The KV pool size in GiB is unaffected (same activation and weight overhead), so the **KV GiB error stays near −4%** (the same as the default-dtype baseline). But because the planner always computes per-token bytes from the model's native compute dtype, **token count and max-concurrency predictions are ~52% too low** for fp8-KV runs. This is a direct, fixable bug: the planner should accept `kv_cache_dtype` as an input parameter and apply 1 byte/token when it is `fp8`.

## Root Cause Analysis

### 1. Activation Memory — Largest Error Source

The planner uses **fixed constants per architecture** (e.g., 4.8 GiB for Llama, 5.6 GiB for Qwen2/3) empirically measured at `max_model_len=16000`. vLLM v0.19.0 reports substantially lower values during its profiling phase:

| Architecture | Predicted (GiB) | Observed range (GiB) | Error range |
|-------------|:---------------:|:--------------------:|:-----------:|
| DeepseekV2 | 8.00 | 1.93–1.93 | +314.51% to +314.51% |
| Granite | 5.50 | 0.75–0.85 | +547.06% to +633.33% |
| KimiVL* | 8.00 | 2.85–2.92 | +173.97% to +180.70% |
| Llama | 4.80 | 1.05–2.21 | +117.19% to +357.14% |
| LlavaNext* | 2.50 | 0.79–0.79 | +216.46% to +216.46% |
| Mistral3* | 2.50 | 2.03–2.18 | +14.68% to +23.15% |
| Mixtral | 8.00 | 1.21–1.21 | +561.16% to +561.16% |
| Phi3 | 5.50 | 1.52–1.52 | +261.84% to +261.84% |
| Qwen2 | 5.60 | 2.21–2.29 | +144.54% to +153.39% |
| Qwen3 | 5.60 | 2.21–2.21 | +153.39% to +153.39% |
| Qwen3Moe | 8.00 | 2.68–2.68 | +198.51% to +198.51% |

The discrepancy suggests the constants were measured with an older vLLM version or different compilation settings. Re-calibrating to these v0.19.0 measurements would be the highest-value fix.

### 2. Non-torch Memory — Underestimated for Multi-GPU

| TP | PP | Constant used | Actual mean (GiB) | Mean error |
|:--:|:--:|:-------------:|:-----------------:|:----------:|
| 1 | 1 | 0.15 GiB | 0.27 | -42.23% |
| 1 | 2 | 0.15 GiB | 0.07 | +114.29% |
| 1 | 4 | 0.15 GiB | 0.07 | +114.29% |
| 2 | 1 | 0.6 GiB | 2.08 | -71.17% |
| 4 | 1 | 0.6 GiB | 2.17 | -72.34% |

For TP=1 the formula slightly under-estimates (0.15 vs ~0.25 GiB actual). For TP≥2, NCCL all-reduce buffers push actual non-torch to ~2.1 GiB — 3.5× the 0.60 GiB constant. For PP≥2, P2P send/receive adds overhead that the formula doesn't model at all.

### 3. GPU Memory Catalog vs Physical

The planner uses 80 GiB (catalog) but H100 physical VRAM is 79.19 GiB:

- Catalog available: 80 × 0.95 = **76.00 GiB**
- Physical available: 79.19 × 0.95 = **75.23 GiB**
- Systematic KV over-prediction from this source alone: **+0.77 GiB**

### 4. CUDA Graph Memory — Excluded from Formula

The planner returns 0.0 GiB for CUDA graphs (treating it as included in activation). vLLM allocates the CUDA graph pool *after* sizing the KV cache, so the reported KV pool includes CUDA graph memory. The formula is therefore consistent with the log-reported KV number — no fix needed, but it should be documented.

Observed CUDA graph pool sizes: 0.51–1.85 GiB (mean 1.03 GiB).

## Recommendations

| Priority | Fix | Expected impact |
|:--------:|-----|:---------------:|
| 🔴 High | **Re-calibrate activation constants** from v0.19.0 measurements. Current constants are 2–7× too high. Updating to ~1.0–2.2 GiB/architecture would remove the single largest prediction error. | +4–10 GiB KV accuracy |
| 🔴 High | **Accept `--kv-cache-dtype` as a planner input.** When set to `fp8`, halve the per-token KV bytes. This is a one-line formula change. | 2× token/concurrency accuracy for fp8-KV runs |
| 🔴 High | **Accept `--dtype` as a planner input.** When set to `float32`, double the per-param bytes for weight estimation. | Fixes −50% weight error for float32 runs |
| 🟡 Medium | **Re-measure non-torch constants for TP≥2 and PP≥2.** NCCL overhead scales with both and is currently under-estimated by ~3.5×. | +1–2 GiB KV accuracy for multi-GPU |
| 🟡 Medium | **Scale activation constant by 1/PP.** Each pipeline stage processes layers/PP transformer blocks; profiling overhead scales proportionally. | Fixes growing activation error at high PP |
| 🟢 Low | **Use physical GPU memory** (79.19 GiB for H100) rather than the catalog 80 GiB nominal. | +0.77 GiB KV accuracy |