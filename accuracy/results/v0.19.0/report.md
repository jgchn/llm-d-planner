# Memory Validation Report

## Per-component error

| Model | TP | PP | DP | max_len | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | 1 | 1 | 1 | 8192 | -0.6% | +314.5% | -42.3% | -11.5% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 2 | 1 | 1 | 8192 | -0.1% | +144.9% | -71.4% | +5.0% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 4 | 1 | 1 | 8192 | -0.2% | +143.7% | -72.9% | +5.9% |
| RedHatAI/Qwen2.5-7B-Instruct-fp8-dynamic | 1 | 1 | 1 | 8192 | -0.4% | +153.4% | -37.5% | -3.9% |
| ibm-granite/granite-3.1-2b-instruct | 1 | 1 | 1 | 8192 | -0.4% | +633.3% | -67.4% | -5.3% |
| ibm-granite/granite-3.1-8b-instruct | 1 | 1 | 1 | 8192 | -0.2% | +547.1% | -67.4% | -6.0% |
| ibm-granite/granite-3.3-8b-instruct | 1 | 1 | 1 | 8192 | -0.2% | +547.1% | -67.4% | -6.0% |
| ibm-granite/granite-vision-3.3-2b | 1 | 1 | 1 | 8192 | 0.0% | +216.5% | -40.0% | -1.2% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | +76.2% | +154.0% | -40.0% | -13.2% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -50.1% | +117.2% | -40.0% | +31.1% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 2048 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 4096 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 2 | 1 | 8192 | -0.4% | +336.4% | +114.3% | -0.9% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 4 | 1 | 8192 | -12.2% | +357.1% | +114.3% | +1.6% |
| meta-llama/Llama-3.1-8B-Instruct | 2 | 1 | 1 | 8192 | -0.4% | +154.0% | -71.0% | +2.8% |
| meta-llama/Llama-3.1-8B-Instruct | 4 | 1 | 1 | 8192 | -0.8% | +154.0% | -71.8% | +4.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 16384 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 32768 | -0.2% | +154.0% | -40.0% | -3.5% |
| microsoft/phi-4 | 1 | 1 | 1 | 8192 | -0.3% | +261.8% | -40.0% | -6.6% |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 1 | 1 | 1 | 8192 | -0.1% | +23.2% | -40.0% | +1.6% |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2 | 1 | 1 | 8192 | -0.0% | +561.2% | -71.0% | -1.9% |
| moonshotai/Kimi-Dev-72B | 2 | 1 | 1 | 8192 | -0.2% | +144.5% | -71.3% | +61.9% |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | -0.4% | +144.5% | -72.9% | +9.3% |
| moonshotai/Kimi-VL-A3B-Instruct | 1 | 1 | 1 | 8192 | -0.6% | +174.0% | -40.0% | -9.8% |
| moonshotai/Kimi-VL-A3B-Instruct | 2 | 1 | 1 | 8192 | -1.6% | +180.7% | -71.0% | +2.4% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | -0.4% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-72B-Instruct | 2 | 1 | 1 | 8192 | -0.1% | +144.5% | -71.3% | +60.9% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | -0.4% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 16384 | -0.4% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 32768 | -0.4% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 2048 | -0.4% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 4096 | -0.4% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | -0.4% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 2 | 1 | 1 | 8192 | -0.4% | +153.4% | -70.9% | +2.6% |
| Qwen/Qwen2.5-7B-Instruct | 4 | 1 | 1 | 8192 | 0.0% | +153.4% | -71.8% | +4.6% |
| Qwen/Qwen3-30B-A3B | 1 | 1 | 1 | 8192 | -0.0% | +198.5% | -44.4% | -28.7% |
| Qwen/Qwen3-8B | 1 | 1 | 1 | 8192 | -0.1% | +153.4% | -40.0% | -4.4% |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 2 | 1 | 1 | 8192 | -0.1% | +144.9% | -71.6% | +5.0% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 | 1 | 1 | 1 | 8192 | -0.7% | +154.0% | -40.0% | -3.0% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.2% | +14.7% | -42.3% | +1.2% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 2 | 1 | 1 | 8192 | -0.8% | +23.2% | -71.0% | +5.3% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.4% | +154.0% | -40.0% | -3.1% |
| RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | -0.4% | +153.4% | -40.0% | -3.9% |

## Per-architecture error

_Group by architecture class. Mean and max absolute error per component._

## Argument sensitivity

### max_model_len sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| 2048 | -0.2% | +154.0% | -40.0% | -3.5% |
| 4096 | -0.2% | +154.0% | -40.0% | -3.5% |
| 16384 | -0.2% | +154.0% | -40.0% | -3.5% |
| 32768 | -0.2% | +154.0% | -40.0% | -3.5% |
| 16384 | -0.4% | +153.4% | -37.5% | -4.2% |
| 32768 | -0.4% | +153.4% | -37.5% | -4.2% |
| 2048 | -0.4% | +153.4% | -37.5% | -4.2% |
| 4096 | -0.4% | +153.4% | -37.5% | -4.2% |

### tp sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| 2 | -0.4% | +154.0% | -71.0% | +2.8% |
| 4 | -0.8% | +154.0% | -71.8% | +4.5% |
| 4 | -0.4% | +144.5% | -72.9% | +9.3% |
| 2 | -0.4% | +153.4% | -70.9% | +2.6% |
| 4 | 0.0% | +153.4% | -71.8% | +4.6% |

### pp sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| 2 | -0.4% | +336.4% | +114.3% | -0.9% |
| 4 | -12.2% | +357.1% | +114.3% | +1.6% |

### dp sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| 1 | -0.2% | +154.0% | -40.0% | -3.5% |

### dtype sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| bfloat16 | -0.2% | +154.0% | -40.0% | -3.5% |
| float16 | -0.2% | +154.0% | -40.0% | -3.5% |
| float32 | -50.1% | +117.2% | -40.0% | +31.1% |

### quantization sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| None | -0.1% | +144.9% | -71.4% | +5.0% |
| None | -0.2% | +143.7% | -72.9% | +5.9% |
| None | -0.4% | +153.4% | -37.5% | -3.9% |
| fp8 | +76.2% | +154.0% | -40.0% | -13.2% |
| None | -0.1% | +144.9% | -71.6% | +5.0% |
| None | -0.7% | +154.0% | -40.0% | -3.0% |
| None | -0.2% | +14.7% | -42.3% | +1.2% |
| None | -0.8% | +23.2% | -71.0% | +5.3% |
| None | -0.4% | +154.0% | -40.0% | -3.1% |
| None | -0.4% | +153.4% | -40.0% | -3.9% |

### kv_cache_dtype sweep

| Value | Weight | Activation | Non-torch | KV cache |
|---|---|---|---|---|
| fp8 | -0.2% | +154.0% | -40.0% | -3.5% |
| fp8 | -0.4% | +153.4% | -37.5% | -4.2% |
| auto | -0.4% | +153.4% | -37.5% | -4.2% |

## Outliers

- **deepseek-ai/DeepSeek-V2-Lite-Chat** (TP=1): {'activation_memory': 314.5077720207254, 'non_torch_memory': -42.307692307692314, 'kv_cache': -11.511121302453557} — root cause required
- **RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic** (TP=2): {'activation_memory': 144.89795918367346, 'non_torch_memory': -71.42857142857143} — root cause required
- **RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic** (TP=4): {'activation_memory': 143.6548223350254, 'non_torch_memory': -72.85067873303167} — root cause required
- **RedHatAI/Qwen2.5-7B-Instruct-fp8-dynamic** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **ibm-granite/granite-3.1-2b-instruct** (TP=1): {'activation_memory': 633.3333333333333, 'non_torch_memory': -67.3913043478261} — root cause required
- **ibm-granite/granite-3.1-8b-instruct** (TP=1): {'activation_memory': 547.0588235294118, 'non_torch_memory': -67.3913043478261} — root cause required
- **ibm-granite/granite-3.3-8b-instruct** (TP=1): {'activation_memory': 547.0588235294118, 'non_torch_memory': -67.3913043478261} — root cause required
- **ibm-granite/granite-vision-3.3-2b** (TP=1): {'activation_memory': 216.45569620253164, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'weight_memory': 76.2073027090695, 'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0, 'kv_cache': -13.18681318681318} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'weight_memory': -50.100066711140755, 'activation_memory': 117.19457013574662, 'non_torch_memory': -40.0, 'kv_cache': 31.051401869158894} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 336.3636363636363, 'non_torch_memory': 114.28571428571426} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'weight_memory': -12.206572769953041, 'activation_memory': 357.1428571428571, 'non_torch_memory': 114.28571428571426} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=2): {'activation_memory': 153.96825396825398, 'non_torch_memory': -71.01449275362319} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=4): {'activation_memory': 153.96825396825398, 'non_torch_memory': -71.83098591549295} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **meta-llama/Llama-3.1-8B-Instruct** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **microsoft/phi-4** (TP=1): {'activation_memory': 261.84210526315786, 'non_torch_memory': -40.0} — root cause required
- **mistralai/Mistral-Small-3.1-24B-Instruct-2503** (TP=1): {'activation_memory': 23.15270935960592, 'non_torch_memory': -40.0} — root cause required
- **mistralai/Mixtral-8x7B-Instruct-v0.1** (TP=2): {'activation_memory': 561.1570247933885, 'non_torch_memory': -71.01449275362319} — root cause required
- **moonshotai/Kimi-Dev-72B** (TP=2): {'activation_memory': 144.54148471615719, 'non_torch_memory': -71.29186602870813, 'kv_cache': 61.920529801324484} — root cause required
- **moonshotai/Kimi-Dev-72B** (TP=4): {'activation_memory': 144.54148471615719, 'non_torch_memory': -72.85067873303167} — root cause required
- **moonshotai/Kimi-VL-A3B-Instruct** (TP=1): {'activation_memory': 173.97260273972603, 'non_torch_memory': -40.0} — root cause required
- **moonshotai/Kimi-VL-A3B-Instruct** (TP=2): {'activation_memory': 180.70175438596493, 'non_torch_memory': -71.01449275362319} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **Qwen/Qwen2.5-72B-Instruct** (TP=2): {'activation_memory': 144.54148471615719, 'non_torch_memory': -71.29186602870813, 'kv_cache': 60.855263157894726} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -37.5} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=2): {'activation_memory': 153.39366515837102, 'non_torch_memory': -70.87378640776699} — root cause required
- **Qwen/Qwen2.5-7B-Instruct** (TP=4): {'activation_memory': 153.39366515837102, 'non_torch_memory': -71.83098591549295} — root cause required
- **Qwen/Qwen3-30B-A3B** (TP=1): {'activation_memory': 198.50746268656715, 'non_torch_memory': -44.44444444444445, 'kv_cache': -28.747566515249833} — root cause required
- **Qwen/Qwen3-8B** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -40.0} — root cause required
- **redhatai/Llama-3.3-70B-Instruct-quantized.w8a8** (TP=2): {'activation_memory': 144.89795918367346, 'non_torch_memory': -71.56398104265402} — root cause required
- **RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8** (TP=1): {'activation_memory': 14.6788990825688, 'non_torch_memory': -42.307692307692314} — root cause required
- **RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8** (TP=2): {'activation_memory': 23.15270935960592, 'non_torch_memory': -71.01449275362319} — root cause required
- **RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8** (TP=1): {'activation_memory': 153.96825396825398, 'non_torch_memory': -40.0} — root cause required
- **RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8** (TP=1): {'activation_memory': 153.39366515837102, 'non_torch_memory': -40.0} — root cause required

## Calibration decisions

_Document constant changes here: old value → new value, evidence._
