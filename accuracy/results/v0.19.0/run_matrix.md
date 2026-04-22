# Run Matrix — vLLM v0.19.0 / H100-80GB

**47 successful runs, 11 failed runs.**

Quantization abbreviations: `ct` = compressed-tensors, `gptq` = gptq_marlin, `fp8` = fp8 inline, `—` = none.

## Successful Runs

| Model | TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +314.5% | -42.3% | -11.5% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.1% | +144.9% | -71.4% | +5.0% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 4 | 1 | 1 | 8192 | bf16 | ct | auto | -0.2% | +143.7% | -72.9% | +5.9% |
| RedHatAI/Qwen2.5-7B-Instruct-fp8-dynamic | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.4% | +153.4% | -37.5% | -3.9% |
| ibm-granite/granite-3.1-2b-instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +633.3% | -67.4% | -5.3% |
| ibm-granite/granite-3.1-8b-instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +547.1% | -67.4% | -6.0% |
| ibm-granite/granite-3.3-8b-instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +547.1% | -67.4% | -6.0% |
| ibm-granite/granite-vision-3.3-2b | 1 | 1 | 1 | 8192 | bf16 | — | auto | +0.0% | +216.5% | -40.0% | -1.2% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | fp8 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | bf16 | fp8 | auto | +76.2% | +154.0% | -40.0% | -13.2% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | f16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | f32 | — | auto | -50.1% | +117.2% | -40.0% | +31.1% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 2048 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 4096 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 16384 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 32768 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 2 | 1 | 8192 | bf16 | — | auto | -0.4% | +336.4% | +114.3% | -0.9% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 4 | 1 | 8192 | bf16 | — | auto | -12.2% | +357.1% | +114.3% | +1.6% |
| meta-llama/Llama-3.1-8B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +154.0% | -71.0% | +2.8% |
| meta-llama/Llama-3.1-8B-Instruct | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.8% | +154.0% | -71.8% | +4.5% |
| microsoft/phi-4 | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.3% | +261.8% | -40.0% | -6.6% |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +23.2% | -40.0% | +1.5% |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +561.2% | -71.0% | -1.9% |
| moonshotai/Kimi-Dev-72B | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +144.5% | -71.3% | +61.8% |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.5% | +144.5% | -72.9% | +9.3% |
| moonshotai/Kimi-VL-A3B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +174.0% | -40.0% | -9.8% |
| moonshotai/Kimi-VL-A3B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -1.6% | +180.7% | -71.0% | +2.4% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | fp8 | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 2048 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 4096 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 16384 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 32768 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +153.4% | -70.9% | +2.6% |
| Qwen/Qwen2.5-7B-Instruct | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +153.4% | -71.8% | +4.6% |
| Qwen/Qwen2.5-72B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +144.5% | -71.3% | +60.8% |
| Qwen/Qwen3-8B | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +153.4% | -40.0% | -4.4% |
| Qwen/Qwen3-30B-A3B | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +198.5% | -44.4% | -28.8% |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.1% | +144.9% | -71.6% | +5.0% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 | 1 | 1 | 1 | 8192 | f16 | gptq | auto | -0.7% | +154.0% | -40.0% | -3.0% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | f16 | ct | auto | -0.4% | +154.0% | -40.0% | -3.1% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.2% | +14.7% | -42.3% | +1.2% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.8% | +23.2% | -71.0% | +5.3% |
| RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.4% | +153.4% | -40.0% | -3.9% |

## Failed Runs

| Model | TP | PP | DP | max_len | Notes |
|---|---|---|---|---|---|
| google/gemma-2-2b-it | 1 | 1 | 1 | 8192 | |
| google/gemma-2-9b-it | 1 | 1 | 1 | 8192 | |
| google/gemma-2-27b-it | 1 | 1 | 1 | 8192 | |
| google/gemma-3-4b-it | 1 | 1 | 1 | 8192 | |
| google/gemma-3-12b-it | 1 | 1 | 1 | 8192 | |
| google/gemma-3-27b-it | 1 | 1 | 1 | 8192 | |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 2 | 8192 | DP=2 |
| meta-llama/Llama-4-Scout | 4 | 1 | 1 | 8192 | |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | second attempt; TP=2 and TP=4 first attempt succeeded |
| openai/GPT-OSS-20B | 1 | 1 | 1 | 8192 | |
| Qwen/Qwen3-14B | 5 | 1 | 1 | 8192 | TP=5 (non-power-of-2) |

## Calibration decisions

_Document constant changes here: old value → new value, evidence._
