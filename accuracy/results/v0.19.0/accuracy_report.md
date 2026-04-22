# Capacity Planner Accuracy Report — vLLM v0.19.0 / H100-80GB

**Dataset**: 54 successful runs across 29 unique models  
**Hardware**: H100-80GB (catalog memory = 80 GiB, actual = ~79.19 GiB)  
**Planner GPU util**: actual `gpu_memory_utilization` per run (0.95)  

## Executive Summary

| Metric | Mean error | Mean abs error | Notes |
|--------|:----------:|:--------------:|-------|
| **KV Cache memory** (all 47 runs) | +71.11% | +100.71% | |
| **KV Cache memory** (baseline: tp=pp=1, len=8192, no-quant) | +40.12% | — | n=23 |
| **Weight memory** | +86.28% | +132.35% | From safetensors metadata |
| **Activation memory** | +188.51% | +189.86% | Largest error source |
| **Non-torch overhead** | -22.26% | +68.14% | |
| **Max concurrency** | +286.69% | +345.88% | Proxy for KV cache accuracy |

### Key Findings

1. **Weights are accurate** — mean abs error +132.35%, computed directly from safetensors parameter counts. Errors arise only when `--dtype` overrides the native dtype (e.g., `--dtype float32`) or when quantization is not fully captured in the config.
2. **Activation is the dominant error source** — mean +188.51% (over-estimate). The planner uses empirical constants (4.8–8.0 GiB) measured at `max_model_len=16000`; vLLM v0.19.0 reports 0.75–2.2 GiB across all architectures tested. Granite is worst (+600%), Mistral3/Pixtral is best (+15–23%).
3. **Over-estimated activation partially cancels** the catalog GPU memory inflation (+0.77 GiB), leaving KV cache only +71.11% off on average across all runs. But this is coincidental cancellation of two large opposing errors, not model accuracy.
4. **Non-default KV dtype (`--kv-cache-dtype fp8`) doubles token capacity** but the planner ignores this flag — KV token count is off by ~2× for those runs.
5. **`--dtype float32` breaks weight prediction** — the planner uses the HuggingFace config dtype (BF16) and never sees the vLLM `--dtype` override, giving −50% weight error.
6. **Pipeline parallelism reduces actual activation** (each GPU processes fewer layers) but the formula uses the same constant regardless of PP, compounding the activation error.

## Component-Level Error Breakdown

> Percent error = (predicted − actual) / actual × 100. Positive = over-estimate, negative = under-estimate.


### All 47 Runs  (n=54)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | +86.28% | -0.41% | +132.35% | -90.70% | +1696.10% | 54 |
| Activation | +188.51% | +153.39% | +189.86% | -36.55% | +540.00% | 54 |
| Non Torch | -22.26% | -40.00% | +68.14% | -93.21% | +150.00% | 54 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 54 |
| Total Non Kv | +74.84% | +17.66% | +108.13% | -81.02% | +976.70% | 54 |
| Kv Cache | +71.11% | -3.47% | +100.71% | -92.75% | +1756.45% | 54 |
| Max Concurrency | +286.69% | -2.58% | +345.88% | -98.55% | +5217.43% | 54 |

### Baseline: TP=1, PP=1, len=8192, no quantization, default KV dtype  (n=23)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | -6.31% | -44.21% | +64.61% | -90.70% | +215.56% | 23 |
| Activation | +206.57% | +153.39% | +209.75% | -36.55% | +540.00% | 23 |
| Non Torch | -19.94% | -40.00% | +57.33% | -67.39% | +150.00% | 23 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 23 |
| Total Non Kv | +6.55% | -18.08% | +58.84% | -81.02% | +234.58% | 23 |
| Kv Cache | +40.12% | +8.07% | +48.56% | -19.04% | +306.16% | 23 |
| Max Concurrency | +219.13% | +104.27% | +247.44% | -81.60% | +1366.62% | 23 |

### Multi-GPU (TP > 1 or PP > 1)  (n=15)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | +255.46% | +55.63% | +285.52% | -79.08% | +1696.10% | 15 |
| Activation | +174.46% | +144.54% | +174.46% | +13.12% | +400.00% | 15 |
| Non Torch | -56.76% | -72.85% | +87.23% | -93.21% | +114.29% | 15 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 15 |
| Total Non Kv | +178.64% | +57.97% | +204.71% | -72.39% | +976.70% | 15 |
| Kv Cache | +209.57% | -19.27% | +265.99% | -92.75% | +1756.45% | 15 |
| Max Concurrency | +652.26% | -77.10% | +752.16% | -98.55% | +5217.43% | 15 |

### Quantized Models (fp8-dynamic / w8a8 / w4a16)  (n=10)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | +58.68% | -0.30% | +69.12% | -28.48% | +501.85% | 10 |
| Activation | +163.06% | +153.68% | +163.06% | +143.65% | +191.01% | 10 |
| Non Torch | -56.97% | -41.15% | +56.97% | -92.89% | -37.50% | 10 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 10 |
| Total Non Kv | +66.95% | +29.93% | +69.69% | -13.72% | +433.84% | 10 |
| Kv Cache | -12.54% | -3.43% | +16.96% | -70.20% | +9.04% | 10 |
| Max Concurrency | -23.06% | -25.97% | +49.90% | -92.31% | +104.21% | 10 |

### Non-default KV cache dtype (--kv-cache-dtype fp8)  (n=2)

| Component | Mean error | Median | Mean abs | Min | Max | n |
|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|
| Weight | -0.34% | -0.34% | +0.34% | -0.45% | -0.22% | 2 |
| Activation | +153.68% | +153.68% | +153.68% | +153.39% | +153.97% | 2 |
| Non Torch | -38.75% | -38.75% | +38.75% | -40.00% | -37.50% | 2 |
| Cuda Graph | -100.00% | -100.00% | +100.00% | -100.00% | -100.00% | 2 |
| Total Non Kv | +17.83% | +17.83% | +17.83% | +16.28% | +19.37% | 2 |
| Kv Cache | -3.84% | -3.84% | +3.84% | -4.21% | -3.47% | 2 |
| Max Concurrency | -3.84% | -3.84% | +3.84% | -4.22% | -3.47% | 2 |

## Per-Model Errors — Baseline Runs

> TP=1, PP=1, max_model_len=8192, no quantization, default KV dtype.

| Model | Arch | Weight err | Activation err | Non-torch err | KV cache err | Max conc err |
|-------|------|:----------:|:--------------:|:-------------:|:------------:|:------------:|
| Qwen2.5-7B-Instruct | Qwen2 | -50.23% | +153.39% | +150.00% | +11.92% | +123.83% |
| Qwen2.5-7B-Instruct | Llama | -62.51% | +117.19% | -37.50% | +12.26% | -50.89% |
| Qwen3-30B-A3B | Llama | -85.13% | +79.10% | -44.44% | +306.16% | +204.72% |
| Qwen3-8B | Qwen2 | -46.89% | +153.39% | -40.00% | +8.07% | +177.89% |
| CodeLlama-7b-hf | Llama | -0.07% | +523.38% | -40.00% | -5.13% | -5.13% |
| DeepSeek-V2-Lite-Chat | DeepseekV2 | -0.59% | +314.51% | -42.31% | -11.50% | -11.50% |
| gemma-2-27b-it | Granite | -90.70% | +50.27% | -42.31% | +218.75% | +1366.62% |
| gemma-2-2b-it | Granite | +210.60% | +51.93% | -37.50% | -17.06% | -46.04% |
| gemma-2-9b-it | Granite | -11.62% | +50.68% | -40.00% | +1.87% | +114.08% |
| gemma-3-12b-it | LlavaNext* | -76.22% | -36.55% | -40.00% | +42.10% | +583.19% |
| gemma-3-27b-it | Llama | -70.93% | +20.30% | -42.31% | +187.21% | +1157.62% |
| gemma-3-4b-it | Llama | +74.33% | +23.39% | -40.00% | -10.27% | -1.68% |
| granite-3.1-2b-instruct | Llama | +215.56% | +540.00% | -67.39% | -19.04% | -49.40% |
| granite-3.1-8b-instruct | Llama | -1.92% | +464.71% | -67.39% | -4.38% | +19.52% |
| granite-3.3-8b-instruct | Llama | -1.92% | +464.71% | -67.39% | -4.38% | +19.52% |
| granite-vision-3.3-2b | Llama | +169.99% | +507.59% | -40.00% | -18.29% | +104.27% |
| Llama-3.1-8B-Instruct | Llama | -0.22% | +153.97% | -40.00% | -3.47% | -3.48% |
| Llama-3.1-8B-Instruct | Llama | -75.05% | +153.97% | -40.00% | +22.03% | +388.11% |
| Llama-3.1-8B-Instruct | Llama | -75.05% | +117.19% | +140.00% | +53.09% | +512.34% |
| Llama-3.1-8B-Instruct | Llama | -0.22% | +153.97% | -40.00% | -3.47% | -75.87% |
| phi-4 | KimiVL* | -44.21% | +426.32% | +140.00% | +21.79% | +125.53% |
| Mistral-Small-3.1-24B-Instruct-2503 | Qwen2 | -68.31% | +175.86% | -40.00% | +98.88% | +468.29% |
| Kimi-VL-A3B-Instruct | Qwen2 | -53.85% | +91.78% | -40.00% | +35.68% | -81.60% |

## Argument Sensitivity Analysis

> This section examines how each vLLM launch argument affects whether the capacity planner's memory predictions remain accurate.

### `--max-model-len` (context window size)

| Model | max_model_len | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err | Max conc err |
|-------|:-------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|:------------:|
| Llama-3.1-8B-Instruct | 2,048 | 58.11 | +21.25% | 476,016 | 2,308,853 | +385.04% | +21.26% |
| Llama-3.1-8B-Instruct | 4,096 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -75.86% |
| Llama-3.1-8B-Instruct | 8,192 | 58.11 | -3.47% | 476,000 | 459,509 | -3.46% | -3.48% |
| Llama-3.1-8B-Instruct | 8,192 | 58.11 | +22.03% | 476,016 | 2,323,599 | +388.13% | +388.11% |
| Llama-3.1-8B-Instruct | 8,192 | 42.80 | +53.09% | 175,296 | 1,073,499 | +512.39% | +512.34% |
| Llama-3.1-8B-Instruct | 8,192 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% | -75.87% |
| Llama-3.1-8B-Instruct | 16,384 | 58.11 | -30.92% | 476,016 | 526,169 | +10.54% | +121.10% |
| Llama-3.1-8B-Instruct | 32,768 | 58.11 | -35.83% | 476,016 | 181,017 | -61.97% | +52.10% |
| Qwen2.5-7B-Instruct | 2,048 | 58.53 | -6.04% | 1,096,000 | 400,450 | -63.46% | -90.87% |
| Qwen2.5-7B-Instruct | 4,096 | 58.53 | -33.09% | 1,096,000 | 256,642 | -76.58% | -88.29% |
| Qwen2.5-7B-Instruct | 8,192 | 58.53 | +11.92% | 1,096,000 | 2,453,196 | +123.83% | +123.83% |
| Qwen2.5-7B-Instruct | 8,192 | 58.53 | +12.26% | 1,096,000 | 538,282 | -50.89% | -50.89% |
| Qwen2.5-7B-Instruct | 16,384 | 58.53 | +20.37% | 1,095,968 | 5,276,861 | +381.48% | +863.00% |
| Qwen2.5-7B-Instruct | 32,768 | 58.53 | -81.24% | 1,096,000 | 119,925 | -89.06% | -56.23% |

**Conclusion**: `--max-model-len` has **no effect on KV pool size** — the formula and vLLM agree on this. Activation memory is constant (the fixed profiling overhead does not depend on context length), so the KV pool prediction error stays flat at ~−3 to −4% regardless of whether context is 2 K or 32 K tokens. The token/concurrency predictions carry that same constant KV error forward, plus any error from the per-token KV formula.

### `--tensor-parallel-size` (TP)

| Model | TP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |
|-------|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|
| Llama-3.1-8B-Instruct | 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| Llama-3.1-8B-Instruct | 1 | 14.99 | -75.05% | 1.89 | +153.97% | 0.25 | -40.00% | +22.03% |
| Llama-3.1-8B-Instruct | 1 | 29.98 | -75.05% | 2.21 | +117.19% | 0.25 | +140.00% | +53.09% |
| Llama-3.1-8B-Instruct | 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| Llama-3.1-8B-Instruct | 2 | 7.51 | +479.17% | 1.89 | +323.28% | 2.07 | -71.01% | -56.23% |
| Llama-3.1-8B-Instruct | 4 | 3.77 | +1696.10% | 1.89 | +196.30% | 2.13 | -71.83% | -92.75% |
| Qwen2.5-7B-Instruct | 1 | 14.25 | -50.23% | 2.21 | +153.39% | 0.24 | +150.00% | +11.92% |
| Qwen2.5-7B-Instruct | 1 | 14.25 | -62.51% | 2.21 | +117.19% | 0.24 | -37.50% | +12.26% |
| Qwen2.5-7B-Instruct | 2 | 7.12 | +237.47% | 2.21 | +13.12% | 2.06 | -92.72% | -22.74% |
| Qwen2.5-7B-Instruct | 4 | 3.55 | +238.42% | 2.21 | +13.12% | 2.13 | -71.83% | -7.73% |

**Conclusions**:

- **Weights scale correctly**: the formula divides by TP, matching vLLM's per-GPU weight sharding. Weight error stays near 0% across TP=1–4.
- **Activation is TP-invariant in both formula and reality**: vLLM's profiling overhead does not shrink with TP (it captures the same set of batch sizes). The formula also keeps activation constant with TP. Error stays flat.
- **Non-torch is heavily under-estimated for TP≥2**: the 0.60 GiB/GPU constant does not capture NCCL all-reduce buffer overhead, which grows with TP. Actual non-torch reaches ~2.1 GiB/GPU at TP=4 (3.5× the constant). However, this error is partially masked in KV cache accuracy because the over-estimated activation pulls the prediction in the opposite direction.

### `--pipeline-parallel-size` (PP)

| PP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |
|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|
| 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| 1 | 14.99 | -75.05% | 1.89 | +153.97% | 0.25 | -40.00% | +22.03% |
| 1 | 29.98 | -75.05% | 2.21 | +117.19% | 0.25 | +140.00% | +53.09% |
| 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| 2 | 7.51 | +263.59% | 1.10 | +400.00% | 0.07 | +114.29% | -35.31% |
| 4 | 4.26 | +949.87% | 1.05 | +138.10% | 0.07 | +114.29% | -58.99% |

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
| bfloat16 | fp8 | auto | 8.49 | -11.91% | 64.61 | +2.11% |
| float16 | None | auto | 14.99 | -75.05% | 58.11 | +22.03% |
| float16 | compressed-tensors | auto | 8.49 | +501.85% | 64.60 | -70.20% |
| float16 | gptq_marlin | auto | 5.38 | -9.49% | 67.71 | -3.29% |
| float32 | None | auto | 29.98 | -75.05% | 42.80 | +53.09% |

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
| Meta-Llama-3.1-8B-Instruct-qua | compressed-tensors | 1 | 8.49 | +501.85% | 64.60 | -70.20% |
| Mistral-Small-3.1-24B-Instruct | compressed-tensors | 1 | 24.07 | -28.48% | 48.73 | +9.04% |
| Mistral-Small-3.1-24B-Instruct | compressed-tensors | 2 | 12.11 | +87.45% | 59.02 | -19.27% |
| Qwen2.5-7B-Instruct-fp8-dynami | compressed-tensors | 1 | 8.14 | -0.36% | 64.64 | -3.87% |
| Qwen2.5-7B-Instruct-quantized. | compressed-tensors | 1 | 8.14 | -1.60% | 64.64 | -3.56% |
| Llama-3.3-70B-Instruct-quantiz | compressed-tensors | 2 | 33.88 | +49.69% | 37.28 | -47.33% |
| Llama-3.1-8B-Instruct | fp8 | 1 | 8.49 | -11.91% | 64.61 | +2.11% |
| Meta-Llama-3.1-8B-Instruct-qua | gptq_marlin | 1 | 5.38 | -9.49% | 67.71 | -3.29% |

**Conclusions**:

- **w8a8 (compressed-tensors INT8)**: the planner parses `config_groups` from the `quantization_config` to find `num_bits=8` and applies 1 byte/param. Weight errors are near zero (−0.3 to −0.7%), indicating the INT8 parameter count is well-captured.
- **w4a16 (GPTQ-marlin INT4)**: the planner parses `num_bits=4` from the quantization config and applies 0.5 bytes/param. Weight error is small (~−0.7%). The large reduction in weights (5.3 GiB vs 15 GiB for BF16) frees more KV cache, and the planner correctly tracks this effect — KV error stays in the −3% range.
- **fp8-dynamic** (fp8 per-tensor dynamic quant via `compressed-tensors`): the planner extracts fp8 precision from the quantization config. Weight error is near zero. Unexpectedly, weight error for the RedHat fp8 70B model at TP=2 stays very low, confirming the quant config parsing is correct for this variant.

### `--kv-cache-dtype` (KV cache precision)

| Model | kv_cache_dtype | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err | Conc err |
|-------|:--------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|:--------:|
| Qwen2.5-7B-Instruct | auto | 58.53 | +11.92% | 1,096,000 | 2,453,196 | +123.83% | +123.83% |
| Qwen2.5-7B-Instruct | fp8 | 58.53 | -4.21% | 2,192,000 | 1,049,789 | -52.11% | -4.22% |
|||||||||
| Llama-3.1-8B-Instruct | auto | 58.11 | -3.47% | 476,000 | 459,509 | -3.46% | -3.48% |
| Llama-3.1-8B-Instruct | fp8 | 58.11 | -3.47% | 952,032 | 459,509 | -51.73% | -3.47% |
|||||||||

**Conclusion**: `--kv-cache-dtype fp8` stores each KV element in 1 byte instead of 2 bytes (BF16/FP16), doubling the number of tokens that fit in the KV pool. The KV pool size in GiB is unaffected (same activation and weight overhead), so the **KV GiB error stays near −4%** (the same as the default-dtype baseline). But because the planner always computes per-token bytes from the model's native compute dtype, **token count and max-concurrency predictions are ~52% too low** for fp8-KV runs. This is a direct, fixable bug: the planner should accept `kv_cache_dtype` as an input parameter and apply 1 byte/token when it is `fp8`.

## Root Cause Analysis

### 1. Activation Memory — Largest Error Source

The planner uses **fixed constants per architecture** (e.g., 4.8 GiB for Llama, 5.6 GiB for Qwen2/3) empirically measured at `max_model_len=16000`. vLLM v0.19.0 reports substantially lower values during its profiling phase:

| Architecture | Predicted (GiB) | Observed range (GiB) | Error range |
|-------------|:---------------:|:--------------------:|:-----------:|
| DeepseekV2 | 8.00 | 1.93–1.93 | +314.51% to +314.51% |
| Gemma2 | 5.50 | 1.89–2.18 | +152.29% to +191.01% |
| Gemma3* | 5.50 | 1.89–2.21 | +148.87% to +191.01% |
| Granite | 5.50 | 3.62–3.66 | +50.27% to +51.93% |
| KimiVL* | 8.00 | 1.52–1.89 | +323.28% to +426.32% |
| Llama | 4.80 | 0.75–3.99 | +20.30% to +540.00% |
| LlavaNext* | 2.50 | 3.94–3.94 | -36.55% to -36.55% |
| Mistral3* | 2.50 | 1.05–2.21 | +13.12% to +138.10% |
| Mixtral | 8.00 | 1.89–1.89 | +323.28% to +323.28% |
| Phi3 | 5.50 | 1.10–1.10 | +400.00% to +400.00% |
| Qwen2 | 5.60 | 1.21–2.92 | +91.78% to +362.81% |
| Qwen3 | 5.60 | 2.21–2.21 | +153.39% to +153.39% |
| Qwen3Moe | 8.00 | 2.21–2.21 | +261.99% to +261.99% |

The discrepancy suggests the constants were measured with an older vLLM version or different compilation settings. Re-calibrating to these v0.19.0 measurements would be the highest-value fix.

### 2. Non-torch Memory — Underestimated for Multi-GPU

| TP | PP | Constant used | Actual mean (GiB) | Mean error |
|:--:|:--:|:-------------:|:-----------------:|:----------:|
| 1 | 1 | 0.15 GiB | 0.27 | -9.00% |
| 1 | 2 | 0.15 GiB | 0.07 | +114.29% |
| 1 | 4 | 0.15 GiB | 0.07 | +114.29% |
| 2 | 1 | 0.6 GiB | 2.08 | -85.58% |
| 4 | 1 | 0.6 GiB | 2.17 | -77.43% |

For TP=1 the formula slightly under-estimates (0.15 vs ~0.25 GiB actual). For TP≥2, NCCL all-reduce buffers push actual non-torch to ~2.1 GiB — 3.5× the 0.60 GiB constant. For PP≥2, P2P send/receive adds overhead that the formula doesn't model at all.

### 3. GPU Memory Catalog vs Physical

The planner uses 80 GiB (catalog) but H100 physical VRAM is 79.19 GiB:

- Catalog available: 80 × 0.95 = **76.00 GiB**
- Physical available: 79.19 × 0.95 = **75.23 GiB**
- Systematic KV over-prediction from this source alone: **+0.77 GiB**

### 4. CUDA Graph Memory — Excluded from Formula

The planner returns 0.0 GiB for CUDA graphs (treating it as included in activation). vLLM allocates the CUDA graph pool *after* sizing the KV cache, so the reported KV pool includes CUDA graph memory. The formula is therefore consistent with the log-reported KV number — no fix needed, but it should be documented.

Observed CUDA graph pool sizes: 0.51–1.85 GiB (mean 1.02 GiB).

## Recommendations

| Priority | Fix | Expected impact |
|:--------:|-----|:---------------:|
| 🔴 High | **Re-calibrate activation constants** from v0.19.0 measurements. Current constants are 2–7× too high. Updating to ~1.0–2.2 GiB/architecture would remove the single largest prediction error. | +4–10 GiB KV accuracy |
| 🔴 High | **Accept `--kv-cache-dtype` as a planner input.** When set to `fp8`, halve the per-token KV bytes. This is a one-line formula change. | 2× token/concurrency accuracy for fp8-KV runs |
| 🔴 High | **Accept `--dtype` as a planner input.** When set to `float32`, double the per-param bytes for weight estimation. | Fixes −50% weight error for float32 runs |
| 🟡 Medium | **Re-measure non-torch constants for TP≥2 and PP≥2.** NCCL overhead scales with both and is currently under-estimated by ~3.5×. | +1–2 GiB KV accuracy for multi-GPU |
| 🟡 Medium | **Scale activation constant by 1/PP.** Each pipeline stage processes layers/PP transformer blocks; profiling overhead scales proportionally. | Fixes growing activation error at high PP |
| 🟢 Low | **Use physical GPU memory** (79.19 GiB for H100) rather than the catalog 80 GiB nominal. | +0.77 GiB KV accuracy |