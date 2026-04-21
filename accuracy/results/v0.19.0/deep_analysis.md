# Capacity Planner — Deep Accuracy Analysis

_vLLM v0.19.0 · H100-80GB · 64 runs · 23 models_

## Executive Summary

**Runs analyzed**: 64 across 23 models on 1 GPU type(s).

### Overall accuracy

| Cohort | N | Mean err | MAE | Min / Max | ≤5% | ≤10% |
|---|---|---|---|---|---|---|
| Weight memory | 64 | -1.8% | +1.8% | -50.1% / 0.0% | 94% | 95% |
| KV cache memory | 64 | +1.9% | +7.6% | -32.7% / +61.9% | 66% | 89% |

## By architecture type

| Cohort | N | Mean err | MAE | Min / Max | ≤5% | ≤10% |
|---|---|---|---|---|---|---|
| **Dense** — weight | 49 | -1.8% | +1.8% | -50.1% / 0.0% | 94% | 96% |
| **Dense** — KV | 49 | +0.9% | +6.2% | -32.7% / +60.9% | 71% | 94% |
| **MoE** — weight | 11 | -2.0% | +2.0% | -11.8% / -0.0% | 91% | 91% |
| **MoE** — KV | 11 | +7.5% | +15.2% | -28.7% / +61.9% | 36% | 64% |
| **Multimodal** — weight | 4 | -0.7% | +0.7% | -1.6% / 0.0% | 100% | 100% |
| **Multimodal** — KV | 4 | -1.5% | +4.0% | -9.8% / +2.6% | 75% | 100% |

## Per-model-family accuracy

| Cohort | N | Mean err | MAE | Min / Max | ≤5% | ≤10% |
|---|---|---|---|---|---|---|
| **DeepSeek** — weight | 3 | -1.5% | +1.5% | -2.7% / -0.6% | 100% | 100% |
| **DeepSeek** — KV | 3 | -2.4% | +5.3% | -11.5% / +3.7% | 67% | 67% |
| **GPT-OSS (openai)** — weight | 1 | -11.8% | +11.8% | -11.8% / -11.8% | 0% | 0% |
| **GPT-OSS (openai)** — KV | 1 | +5.5% | +5.5% | +5.5% / +5.5% | 0% | 100% |
| **Granite** — weight | 6 | -0.9% | +0.9% | -1.8% / -0.2% | 100% | 100% |
| **Granite** — KV | 6 | -0.8% | +2.9% | -6.0% / +2.7% | 67% | 100% |
| **Granite-Vision** — weight | 2 | -0.4% | +0.4% | -0.7% / 0.0% | 100% | 100% |
| **Granite-Vision** — KV | 2 | +0.7% | +1.9% | -1.2% / +2.6% | 100% | 100% |
| **Kimi** — weight | 2 | -0.3% | +0.3% | -0.4% / -0.2% | 100% | 100% |
| **Kimi** — KV | 2 | +35.6% | +35.6% | +9.3% / +61.9% | 0% | 50% |
| **Kimi-VL** — weight | 2 | -1.1% | +1.1% | -1.6% / -0.6% | 100% | 100% |
| **Kimi-VL** — KV | 2 | -3.7% | +6.1% | -9.8% / +2.4% | 50% | 100% |
| **Llama-3.1** — weight | 16 | -4.2% | +4.2% | -50.1% / -0.2% | 88% | 88% |
| **Llama-3.1** — KV | 16 | +0.2% | +4.8% | -3.5% / +31.1% | 94% | 94% |
| **Llama-3.3** — weight | 5 | -0.2% | +0.2% | -0.2% / -0.1% | 100% | 100% |
| **Llama-3.3** — KV | 5 | -2.2% | +10.9% | -32.7% / +5.9% | 0% | 80% |
| **Llama-4** — weight | 1 | -4.8% | +4.8% | -4.8% / -4.8% | 100% | 100% |
| **Llama-4** — KV | 1 | +36.2% | +36.2% | +36.2% / +36.2% | 0% | 0% |
| **Mistral-Small** — weight | 5 | -1.7% | +1.7% | -5.7% / -0.1% | 80% | 100% |
| **Mistral-Small** — KV | 5 | +4.5% | +4.5% | +1.2% / +7.5% | 40% | 100% |
| **Mixtral** — weight | 2 | -0.0% | +0.0% | -0.0% / -0.0% | 100% | 100% |
| **Mixtral** — KV | 2 | +0.3% | +2.2% | -1.9% / +2.4% | 100% | 100% |
| **Phi** — weight | 2 | -0.6% | +0.6% | -0.9% / -0.3% | 100% | 100% |
| **Phi** — KV | 2 | -2.3% | +4.3% | -6.6% / +2.0% | 50% | 100% |
| **Qwen2.5** — weight | 13 | -0.4% | +0.4% | -0.4% / 0.0% | 100% | 100% |
| **Qwen2.5** — KV | 13 | +3.1% | +8.8% | -4.2% / +60.9% | 85% | 92% |
| **Qwen3** — weight | 4 | -0.1% | +0.1% | -0.3% / -0.0% | 100% | 100% |
| **Qwen3** — KV | 4 | -5.8% | +10.8% | -28.7% / +5.4% | 50% | 75% |

## TP sensitivity

_KV cache error grouped by tensor-parallel degree (all models). After applying the per-GPU normalisation (÷TP×PP)._

| Cohort | N | Mean err | MAE | Min / Max | ≤5% | ≤10% |
|---|---|---|---|---|---|---|
| TP=1 | 34 | -4.3% | +6.4% | -32.7% / +31.1% | 76% | 88% |
| TP=2 | 15 | +10.5% | +10.7% | -1.9% / +61.9% | 60% | 87% |
| TP=4 | 15 | +7.3% | +7.3% | +2.4% / +36.2% | 47% | 93% |

## PP sensitivity

_KV cache error grouped by pipeline-parallel degree._

| Cohort | N | Mean err | MAE | Min / Max | ≤5% | ≤10% |
|---|---|---|---|---|---|---|
| PP=1 | 62 | +1.9% | +7.8% | -32.7% / +61.9% | 65% | 89% |
| PP=2 | 1 | -0.9% | +0.9% | -0.9% / -0.9% | 100% | 100% |
| PP=4 | 1 | +1.6% | +1.6% | +1.6% / +1.6% | 100% | 100% |

## Context-length sensitivity (TP=1 runs only)

_Models tested at multiple max_model_len values. KV cache error should be constant if the formula is context-length-agnostic._

**Qwen/Qwen2.5-7B-Instruct**

| max_len | KV err |
|---|---|
| 2048 | -4.2% |
| 4096 | -4.2% |
| 8192 | -4.2% |
| 8192 | -4.2% |
| 8192 | -4.2% |
| 8192 | -4.2% |
| 16384 | -4.2% |
| 32768 | -4.2% |

**meta-llama/Llama-3.1-8B-Instruct**

| max_len | KV err |
|---|---|
| 2048 | -3.5% |
| 4096 | -3.5% |
| 8192 | -3.5% |
| 8192 | -3.5% |
| 8192 | -3.5% |
| 8192 | +31.1% |
| 8192 | -3.5% |
| 32768 | -3.5% |

## Outliers (|error| > 10%)

| Model | TP | PP | Weight err | KV err | Likely cause |
|---|---|---|---|---|---|
| moonshotai/Kimi-Dev-72B | 2 | 1 | -0.2% | +61.9% | TP/PP residual: per-GPU normalisation may be imprecise |
| Qwen/Qwen2.5-72B-Instruct | 2 | 1 | -0.1% | +60.9% | TP/PP residual: per-GPU normalisation may be imprecise |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 4 | 1 | -4.8% | +36.2% | TP/PP residual: per-GPU normalisation may be imprecise |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 1 | 1 | -0.1% | -32.7% | large model: activation constant may underestimate real overhead |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | -50.1% | +31.1% | KV formula overestimates available budget |
| Qwen/Qwen3-30B-A3B | 1 | 1 | -0.0% | -28.7% | MoE: routing overhead not modeled in activation/KV budget |
| deepseek-ai/DeepSeek-V2-Lite-Chat | 1 | 1 | -0.6% | -11.5% | unknown |
| openai/gpt-oss-20b | 4 | 1 | -11.8% | +5.5% | MoE/sparse model: shared expert / embedding memory not sharded by TP |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 4 | -12.2% | +1.6% | PP≥4: weight sharding formula incorrect for high PP |

## Calibration notes

### Weight memory

- Mean error -1.8% — slightly negative (planner underestimates). Cause: safetensors metadata reports storage dtype; actual in-memory size can differ due to alignment/padding.

- PP≥4 and certain MoE models show >10% weight error — embedding and shared-expert tensors may not be sharded by TP/PP as assumed by the formula.

### KV cache memory (TP=1)

- TP=1 KV mean error -4.6% (MAE +6.7%). Mostly within ±10%.

- Consistent negative bias across TP=1 configs suggests activation_memory constant is slightly too high (over-reserves budget, leaving less for KV).

### KV cache memory (TP>1)

- After ÷(TP×PP) normalisation, errors are within ±10% for most models.

- Remaining positive bias at TP=2/4 is consistent with extra NCCL/all-gather buffers not captured by non_torch constant.

### Large-model KV outliers

- `Qwen3-30B-A3B` (TP=1): −29%. MoE routing buffers consume more memory than modeled.

- `Llama-3.3-70B-w8a8` (TP=1): −33%. W8A8 quantization increases activation-memory footprint (dequant workspace) not accounted for in constant.

- `Kimi-Dev-72B` (TP=2): +62%. Likely residual normalisation issue or model-specific memory layout.

- `Qwen2.5-72B` (TP=2): +61%. Same pattern as Kimi-Dev-72B — large model at TP=2 still shows excess after normalisation.

## Per-model breakdown

### Qwen/Qwen2.5-72B-Instruct  _Qwen2.5 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.1% | +60.9% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.4% | +9.3% |

### Qwen/Qwen2.5-7B-Instruct  _Qwen2.5 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 2048 | auto | — | auto | -0.4% | -4.2% |
| 1 | 1 | 1 | 4096 | auto | — | auto | -0.4% | -4.2% |
| 1 | 1 | 1 | 8192 | bfloat16 | — | fp8 | -0.4% | -4.2% |
| 1 | 1 | 1 | 8192 | bfloat16 | — | auto | -0.4% | -4.2% |
| 1 | 1 | 1 | 8192 | float16 | — | auto | -0.4% | -4.2% |
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.4% | -4.2% |
| 1 | 1 | 1 | 16384 | auto | — | auto | -0.4% | -4.2% |
| 1 | 1 | 1 | 32768 | auto | — | auto | -0.4% | -4.2% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.4% | +2.6% |
| 4 | 1 | 1 | 8192 | auto | — | auto | 0.0% | +4.6% |

### Qwen/Qwen3-30B-A3B  _Qwen3 · MoE_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.0% | -28.7% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.2% | +5.4% |

### Qwen/Qwen3-8B  _Qwen3 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.1% | -4.4% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.3% | +4.7% |

### RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic  _Llama-3.3 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.1% | +5.0% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.2% | +5.9% |

### RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16  _Llama-3.1 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | float16 | — | auto | -0.7% | -3.0% |

### RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8  _Llama-3.1 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | float16 | — | auto | -0.4% | -3.1% |
| 1 | 1 | 1 | 8192 | float16 | — | auto | -0.4% | -3.1% |
| 1 | 1 | 1 | 8192 | float16 | — | auto | -0.4% | -3.1% |

### RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8  _Mistral-Small · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.2% | +1.2% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.8% | +5.3% |

### RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8  _Qwen2.5 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.4% | -3.9% |

### deepseek-ai/DeepSeek-V2-Lite-Chat  _DeepSeek · MoE_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.6% | -11.5% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -1.3% | +0.6% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -2.7% | +3.7% |

### ibm-granite/granite-3.1-2b-instruct  _Granite · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.4% | -5.3% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.8% | +0.4% |

### ibm-granite/granite-3.1-8b-instruct  _Granite · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8192 | auto | — | auto | -1.8% | +2.7% |

### ibm-granite/granite-3.3-8b-instruct  _Granite · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.2% | -6.0% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.4% | +0.6% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -1.8% | +2.7% |

### ibm-granite/granite-vision-3.3-2b  _Granite-Vision · Multimodal_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | 0.0% | -1.2% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.7% | +2.6% |

### meta-llama/Llama-3.1-8B-Instruct  _Llama-3.1 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 2048 | auto | — | auto | -0.2% | -3.5% |
| 1 | 1 | 1 | 4096 | auto | — | auto | -0.2% | -3.5% |
| 1 | 1 | 1 | 8192 | bfloat16 | — | fp8 | -0.2% | -3.5% |
| 1 | 1 | 1 | 8192 | bfloat16 | — | auto | -0.2% | -3.5% |
| 1 | 1 | 1 | 8192 | float16 | — | auto | -0.2% | -3.5% |
| 1 | 1 | 1 | 8192 | float32 | — | auto | -50.1% | +31.1% |
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.2% | -3.5% |
| 1 | 1 | 1 | 32768 | auto | — | auto | -0.2% | -3.5% |
| 1 | 2 | 1 | 8192 | auto | — | auto | -0.4% | -0.9% |
| 1 | 4 | 1 | 8192 | auto | — | auto | -12.2% | +1.6% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.4% | +2.8% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.8% | +4.5% |

### meta-llama/Llama-4-Scout-17B-16E-Instruct  _Llama-4 · MoE_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8192 | auto | — | auto | -4.8% | +36.2% |

### microsoft/phi-4  _Phi · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.3% | -6.6% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.9% | +2.0% |

### mistralai/Mistral-Small-3.1-24B-Instruct-2503  _Mistral-Small · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.1% | +1.6% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -1.9% | +7.2% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -5.7% | +7.5% |

### mistralai/Mixtral-8x7B-Instruct-v0.1  _Mixtral · MoE_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.0% | -1.9% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.0% | +2.4% |

### moonshotai/Kimi-Dev-72B  _Kimi · MoE_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.2% | +61.9% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.4% | +9.3% |

### moonshotai/Kimi-VL-A3B-Instruct  _Kimi-VL · Multimodal_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.6% | -9.8% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -1.6% | +2.4% |

### openai/gpt-oss-20b  _GPT-OSS (openai) · MoE_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8192 | auto | — | auto | -11.8% | +5.5% |

### redhatai/Llama-3.3-70B-Instruct-quantized.w8a8  _Llama-3.3 · Dense_

| TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | KV err |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 8192 | auto | — | auto | -0.1% | -32.7% |
| 2 | 1 | 1 | 8192 | auto | — | auto | -0.1% | +5.0% |
| 4 | 1 | 1 | 8192 | auto | — | auto | -0.2% | +5.9% |
