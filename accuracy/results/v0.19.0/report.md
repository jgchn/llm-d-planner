# Memory Validation Report

## Per-component error

| Model | TP | PP | DP | max_len | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | 1 | 1 | 1 | 8192 | -0.6% | — | — | -11.5% |
| deepseek-ai/DeepSeek-V2-Lite-Chat | 2 | 1 | 1 | 8192 | -1.3% | — | — | +0.6% |
| deepseek-ai/DeepSeek-V2-Lite-Chat | 4 | 1 | 1 | 8192 | -2.7% | — | — | +3.7% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 2 | 1 | 1 | 8192 | -0.1% | — | — | +5.0% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 4 | 1 | 1 | 8192 | -0.2% | — | — | +5.9% |
| ibm-granite/granite-3.1-2b-instruct | 1 | 1 | 1 | 8192 | -0.4% | — | — | -5.3% |
| ibm-granite/granite-3.1-2b-instruct | 2 | 1 | 1 | 8192 | -0.8% | — | — | +0.4% |
| ibm-granite/granite-3.1-8b-instruct | 4 | 1 | 1 | 8192 | -1.8% | — | — | +2.7% |
| ibm-granite/granite-3.3-8b-instruct | 1 | 1 | 1 | 8192 | -0.2% | — | — | -6.0% |
| ibm-granite/granite-3.3-8b-instruct | 2 | 1 | 1 | 8192 | -0.4% | — | — | +0.6% |
| ibm-granite/granite-3.3-8b-instruct | 4 | 1 | 1 | 8192 | -1.8% | — | — | +2.7% |
| ibm-granite/granite-vision-3.3-2b | 1 | 1 | 1 | 8192 | 0.0% | — | — | -1.2% |
| ibm-granite/granite-vision-3.3-2b | 2 | 1 | 1 | 8192 | -0.7% | — | — | +2.6% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | — | — | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | — | — | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | — | — | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -50.1% | — | — | +31.1% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 2048 | -0.2% | — | — | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 4096 | -0.2% | — | — | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | — | — | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 2 | 1 | 8192 | -0.4% | — | — | -0.9% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 4 | 1 | 8192 | -12.2% | — | — | +1.6% |
| meta-llama/Llama-3.1-8B-Instruct | 2 | 1 | 1 | 8192 | -0.4% | — | — | +2.8% |
| meta-llama/Llama-3.1-8B-Instruct | 4 | 1 | 1 | 8192 | -0.8% | — | — | +4.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 32768 | -0.2% | — | — | -3.5% |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 4 | 1 | 1 | 8192 | -4.8% | — | — | +36.2% |
| microsoft/phi-4 | 1 | 1 | 1 | 8192 | -0.3% | — | — | -6.6% |
| microsoft/phi-4 | 2 | 1 | 1 | 8192 | -0.9% | — | — | +2.0% |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 1 | 1 | 1 | 8192 | -0.1% | — | — | +1.6% |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 2 | 1 | 1 | 8192 | -1.9% | — | — | +7.2% |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 4 | 1 | 1 | 8192 | -5.7% | — | — | +7.5% |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2 | 1 | 1 | 8192 | -0.0% | — | — | -1.9% |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 4 | 1 | 1 | 8192 | -0.0% | — | — | +2.4% |
| moonshotai/Kimi-Dev-72B | 2 | 1 | 1 | 8192 | -0.2% | — | — | +61.9% |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | -0.4% | — | — | +9.3% |
| moonshotai/Kimi-VL-A3B-Instruct | 1 | 1 | 1 | 8192 | -0.6% | — | — | -9.8% |
| moonshotai/Kimi-VL-A3B-Instruct | 2 | 1 | 1 | 8192 | -1.6% | — | — | +2.4% |
| openai/gpt-oss-20b | 4 | 1 | 1 | 8192 | -11.8% | — | — | +5.5% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-72B-Instruct | 2 | 1 | 1 | 8192 | -0.1% | — | — | +60.9% |
| Qwen/Qwen2.5-72B-Instruct | 4 | 1 | 1 | 8192 | -0.4% | — | — | +9.3% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 16384 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 32768 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 2048 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 4096 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | -0.4% | — | — | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 2 | 1 | 1 | 8192 | -0.4% | — | — | +2.6% |
| Qwen/Qwen2.5-7B-Instruct | 4 | 1 | 1 | 8192 | 0.0% | — | — | +4.6% |
| Qwen/Qwen3-30B-A3B | 1 | 1 | 1 | 8192 | -0.0% | — | — | -28.7% |
| Qwen/Qwen3-30B-A3B | 4 | 1 | 1 | 8192 | -0.2% | — | — | +5.4% |
| Qwen/Qwen3-8B | 1 | 1 | 1 | 8192 | -0.1% | — | — | -4.4% |
| Qwen/Qwen3-8B | 4 | 1 | 1 | 8192 | -0.3% | — | — | +4.7% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.4% | — | — | -3.1% |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.1% | — | — | -32.7% |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 2 | 1 | 1 | 8192 | -0.1% | — | — | +5.0% |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 4 | 1 | 1 | 8192 | -0.2% | — | — | +5.9% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.4% | — | — | -3.1% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 | 1 | 1 | 1 | 8192 | -0.7% | — | — | -3.0% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.2% | — | — | +1.2% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 2 | 1 | 1 | 8192 | -0.8% | — | — | +5.3% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.4% | — | — | -3.1% |
| RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.4% | — | — | -3.9% |

## Per-architecture error

_Group by architecture class. Mean and max absolute error per component._

## Argument sensitivity

### max_model_len sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| 2048 | -0.2% | — | — | -3.5% |
| 4096 | -0.2% | — | — | -3.5% |
| 32768 | -0.2% | — | — | -3.5% |
| 16384 | -0.4% | — | — | -4.2% |
| 32768 | -0.4% | — | — | -4.2% |
| 2048 | -0.4% | — | — | -4.2% |
| 4096 | -0.4% | — | — | -4.2% |

### pp sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| 2 | -0.4% | — | — | -0.9% |
| 4 | -12.2% | — | — | +1.6% |

### dtype sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| bfloat16 | -0.2% | — | — | -3.5% |
| float32 | -50.1% | — | — | +31.1% |

### quantization sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| None | -0.1% | — | — | +5.0% |
| None | -0.2% | — | — | +5.9% |
| None | -0.4% | — | — | -3.1% |
| None | -0.4% | — | — | -3.1% |
| None | -0.7% | — | — | -3.0% |
| None | -0.2% | — | — | +1.2% |
| None | -0.8% | — | — | +5.3% |
| None | -0.4% | — | — | -3.1% |
| None | -0.4% | — | — | -3.9% |

### kv_cache_dtype sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| fp8 | -0.2% | — | — | -3.5% |
| auto | -0.2% | — | — | -3.5% |
| fp8 | -0.4% | — | — | -4.2% |
| auto | -0.4% | — | — | -4.2% |
| auto | -0.4% | — | — | -4.2% |

## Outliers

- **deepseek-ai/DeepSeek-V2-Lite-Chat** (TP=1): {'kv_cache': -11.511121302453557} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'weight_memory': -50.100066711140755, 'kv_cache': 31.051401869158894} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'weight_memory': -12.206572769953041} — root cause required
- **meta-llama/Llama-4-Scout-17B-16E-Instruct** (TP=4): {'kv_cache': 36.179104477611936} — root cause required
- **moonshotai/Kimi-Dev-72B** (TP=2): {'kv_cache': 61.920529801324484} — root cause required
- **openai/gpt-oss-20b** (TP=4): {'weight_memory': -11.845730027548202} — root cause required
- **Qwen/Qwen2.5-72B-Instruct** (TP=2): {'kv_cache': 60.855263157894726} — root cause required
- **Qwen/Qwen3-30B-A3B** (TP=1): {'kv_cache': -28.747566515249833} — root cause required
- **redhatai/Llama-3.3-70B-Instruct-quantized.w8a8** (TP=1): {'kv_cache': -32.73453093812375} — root cause required

## Calibration decisions

_Document constant changes here: old value → new value, evidence._
