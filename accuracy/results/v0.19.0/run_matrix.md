# Run Matrix — vLLM v0.19.0 / H100-80GB

**55 successful runs, 8 failed runs.**

Quantization abbreviations: `ct` = compressed-tensors, `gptq` = gptq_marlin, `fp8` = fp8 inline, `mxfp4` = mx-format fp4, `—` = none.

## Successful Runs

| Model | TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|---|---|---|---|---|---|---|
| codellama/CodeLlama-7b-hf | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +523.4% | -40.0% | -5.1% |
| deepseek-ai/DeepSeek-V2-Lite-Chat | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +314.5% | -42.3% | -11.5% |
| google/gemma-2-27b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +50.3% | -42.3% | -4.6% |
| google/gemma-2-2b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +51.9% | -37.5% | -1.5% |
| google/gemma-2-9b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +50.7% | -40.0% | -1.8% |
| google/gemma-3-12b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -2.6% | +39.6% | -40.0% | -0.1% |
| google/gemma-3-27b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.7% | +37.8% | -42.3% | -1.4% |
| google/gemma-3-4b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -6.6% | +41.4% | -40.0% | -0.3% |
| google/gemma-7b | 1 | 1 | 1 | 8192 | bf16 | — | auto | +0.0% | -34.0% | +66.7% | +1.8% |
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
| meta-llama/Llama-3.1-8B-Instruct | 1 | 2 | 1 | 8192 | bf16 | — | auto | -0.4% | +336.4% | +114.3% | -0.9% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 4 | 1 | 8192 | bf16 | — | auto | -12.2% | +357.1% | +114.3% | +1.6% |
| meta-llama/Llama-3.1-8B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +154.0% | -71.0% | +2.8% |
| meta-llama/Llama-3.1-8B-Instruct | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.8% | +154.0% | -71.8% | +4.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 16384 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 32768 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| microsoft/phi-2 | 1 | 1 | 1 | 2048 | f16 | — | auto | +0.2% | -85.6% | +60.0% | +5.9% |
| microsoft/phi-4 | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.3% | +261.8% | -40.0% | -6.6% |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +23.2% | -40.0% | +1.5% |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +561.2% | -71.0% | -1.9% |
| moonshotai/Kimi-Dev-72B | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +144.5% | -71.3% | +61.8% |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.5% | +144.5% | -72.9% | +9.3% |
| moonshotai/Kimi-VL-A3B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +174.0% | -40.0% | -9.8% |
| moonshotai/Kimi-VL-A3B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -1.6% | +180.7% | -71.0% | +2.4% |
| openai/gpt-oss-20b | 2 | 1 | 1 | 8192 | bf16 | mxfp4 | auto | +9.4% | -64.1% | +245.0% | -2.7% |
| Qwen/Qwen2.5-72B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +144.5% | -71.3% | +60.8% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | fp8 | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 16384 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 32768 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 2048 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 4096 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +153.4% | -70.9% | +2.6% |
| Qwen/Qwen2.5-7B-Instruct | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +153.4% | -71.8% | +4.6% |
| Qwen/Qwen3-30B-A3B | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +198.5% | -44.4% | -28.8% |
| Qwen/Qwen3-8B | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +153.4% | -40.0% | -4.4% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.1% | +144.9% | -71.4% | +5.0% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 4 | 1 | 1 | 8192 | bf16 | ct | auto | -0.2% | +143.7% | -72.9% | +5.9% |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.1% | +144.9% | -71.6% | +5.0% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 | 1 | 1 | 1 | 8192 | f16 | gptq | auto | -0.7% | +154.0% | -40.0% | -3.0% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | f16 | ct | auto | -0.4% | +154.0% | -40.0% | -3.1% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.2% | +14.7% | -42.3% | +1.2% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.8% | +23.2% | -71.0% | +5.3% |
| RedHatAI/Qwen2.5-7B-Instruct-fp8-dynamic | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.4% | +153.4% | -37.5% | -3.9% |
| RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.4% | +153.4% | -40.0% | -3.9% |

## Failed Runs

| Model | TP | PP | DP | max_len | Notes |
|---|---|---|---|---|---|
| codellama/CodeLlama-34b-hf | 2 | 1 | 1 | 8192 | GPU contention at runtime |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 2 | 8192 | DP=2 |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 4 | 1 | 1 | 8192 |  |
| microsoft/phi-2 | 1 | 1 | 1 | 8192 | max_model_len=8192 > max_position_embeddings=2048; fixed with max_model_len=2048 |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | second attempt; tp=2 succeeded |
| openai/gpt-oss-20b | 1 | 1 | 1 | 8192 | sampler warmup OOM (~786 MiB needed, <552 MiB free) |
| openai/gpt-oss-20b | 2 | 1 | 1 | 8192 | sampler warmup OOM at gmu=0.95; succeeded at gmu=0.90 |
| Qwen/Qwen3-14B | 5 | 1 | 1 | 8192 | tp=5 invalid (vocab not divisible by 5) |

## Calibration decisions

_Document constant changes here: old value → new value, evidence._
