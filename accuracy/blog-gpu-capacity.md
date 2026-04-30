# 61 Experiments Later: What We Learned About LLM Memory Prediction

*GPU memory estimation for LLM deployments is still mostly guesswork, and it's the step you have to get right before anything else. Here's what we learned from measuring it empirically across 35 architectures.*

---

You're planning a benchmark suite and need to know how many GPUs each model requires before sizing the cluster. You're launching a serving application and want to avoid over-provisioning by 3x. You're a researcher asking whether two H100s will be enough for a 70B model.

The question is the same: **how much GPU memory will this actually need?**

Memory is the first gate. Either the model fits or it doesn't, and the only way to find out without a prediction tool is to spin up a vLLM server and see if it OOMs. Tensor parallelism, pipeline parallelism, quantization, and long-context windows all change the footprint in non-obvious ways, which makes trial-and-error expensive. This post is about memory estimation specifically: not throughput, latency, or any other performance metric. Those questions are downstream; this one comes first.

[llm-d-planner](https://github.com/llm-d-incubation/llm-d-planner) includes a pip-installable capacity planner that answers this question before you touch a cluster, using only model config files and safetensor headers. To verify it wasn't replacing guesswork with false precision, we ran 61 experiments on H100 GPUs. Here's what we found.

---

## How the Planner Works

Memory breaks into four components: weights, KV cache, activation memory, and non-torch overhead (CUDA runtime + NCCL buffers for multi-GPU). The planner reads `config.json` and safetensor headers for weights, reverse-engineers vLLM's KV cache allocation strategy, and uses empirically measured per-architecture constants for activation memory. No GPU required.

**Known gaps:** fp8 KV cache dtype and runtime fp8 quantization are not yet modeled. These can cut memory by 40-50%, so the planner will over-estimate for those configurations without issuing a warning. Float32 dtype overrides are also unsupported. If you're running fp8-quantized models today, treat the output as a baseline upper bound.

---

## The Experiment

We launched vLLM servers across 61 configurations on H100-80GB GPUs, captured startup logs, and compared measured memory against predictions per component. The sweep covered:

- **35 model architectures**: Llama, Qwen, Gemma, Granite, Mistral, DeepSeek, Phi, Mixtral, multimodal models (LLaVA, Kimi-VL, MiMo)
- **Tensor parallelism** (TP 1, 2, 4) and **pipeline parallelism** (PP 1, 2, 3, 4)
- **Context lengths** from 2,048 to 32,768 tokens
- **Dtype and quantization**: bfloat16, float16; compressed-tensors, GPTQ
- **vLLM version sensitivity**: Qwen3-14B across v0.15.0-v0.19.0

[Raw logs and run JSON files](https://drive.google.com/drive/folders/1a0y2gdhcpKcFxm4RsqXUKWW40Gpd2Kx5) are published; the analysis is reproducible locally without cluster access.

---

## What We Found

**Weight memory: 0.84% mean error** across 49 of the 61 runs (the remaining 12 used fp8/float32 configurations not yet modeled). Weights are the largest single component; for Llama-3.1-8B at TP=1, that's ~15 GiB of 79 GiB available. The formula handles dense, MoE, multimodal, and quantized architectures by reading exact tensor shapes from safetensor headers, making it generalizable to any HuggingFace model beyond the 35 tested.

**KV cache memory: 0.89% mean error** across all runs. This is the component that determines maximum concurrent token budget. One insight worth flagging: the KV pool size is *independent* of `max_model_len`. vLLM sizes the pool from whatever memory remains after weights and activations, then determines how many tokens fit. Setting a longer context window doesn't shrink your pool; it just means each request consumes a larger share, reducing concurrency at max context length.

**Activation memory: +212% mean error**, but in absolute terms that's a ~2.9 GiB over-estimate on a 79 GiB GPU. The more interesting finding is the root cause: **vLLM v0.17.0 reduced activation memory by ~60%, and we didn't notice.**

| vLLM version | Activation (Qwen3-14B) |
|:---:|:---:|
| v0.15.0 | 5.64 GiB |
| v0.16.0 | 5.64 GiB |
| **v0.17.0** | **2.23 GiB** |
| v0.18.0 | 2.23 GiB |
| v0.19.0 | ~2.21 GiB |

Our constants were calibrated against v0.16.0 and never updated. The reduction freed memory that vLLM reallocated to the KV cache, actually improving serving capacity, but our planner was blind to it. Re-calibrating against v0.19.0 measurements is the highest-priority fix; contributions are welcome. The planner is not version-aware for older releases; if you're running an earlier vLLM in production, expect activation estimates to diverge.

**Non-torch overhead** was under-estimated by 44% on average: small at TP=1 (~0.25 GiB actual vs 0.15 GiB predicted), more meaningful at TP>=2 (~2.1 GiB actual vs 0.60 GiB predicted). The sweep also caught a correctness bug: `find_possible_tp` wasn't verifying that TP values divide `vocab_size`, which could cause vLLM to reject configurations the planner suggested as valid. Fixed.

Additional findings from the full data:

- **Max concurrency tracked KV accuracy (3.68% mean error).** The planner predicts how many concurrent requests fit at a given context length (the number most teams actually want), and it inherits the KV cache error directly, since concurrency is just KV tokens divided by `max_model_len`.
- **Activation error varies significantly by architecture.** Granite was +633%, Mistral3 was only +23%. The v0.17.0 reduction wasn't uniform; some architectures were hit harder, and the per-architecture constants need to be re-measured individually.
- **Context length has zero effect on the KV pool size.** KV GiB was identical to two decimal places from 2K to 32K tokens across both Llama and Qwen models, confirming that `max_model_len` controls request size, not pool allocation.
- **Pipeline parallelism introduces weight error even with perfectly divisible layers.** Qwen3-8B has 36 layers; PP=3 gives exactly 12 per stage, yet weight error was -7.20%. The formula assumes all layers are equal size, but embedding and LM-head layers aren't evenly distributed across pipeline stages.
- **Valid TP must divide both `num_attention_heads` and `vocab_size`.** vLLM shards the embedding and LM-head across TP ranks, so a TP value that doesn't divide `vocab_size` will be rejected at startup even if it evenly divides the attention heads. Qwen3-14B has 40 heads (TP=5 looks valid) but `vocab_size=151936` is not divisible by 5, so vLLM rejects it. The planner was only checking attention heads; the fix is to return divisors of `gcd(num_attention_heads, vocab_size)`.
- **`kv_cache_dtype fp8` doubles token capacity but leaves the KV pool size in GiB unchanged.** fp8 halves per-token storage, so twice as many tokens fit in the same memory pool, but the pool size in GiB is unaffected. The planner doesn't yet accept `kv_cache_dtype` as an input, so it under-estimates token count by ~2x for fp8 KV configurations while getting the GiB right.
- **Runtime quantization and dtype overrides cause large weight errors.** `--quantization fp8` compresses weights on-the-fly to ~half the BF16 size, but the planner reads the HuggingFace config (which has no quantization entry) and predicts full BF16 weights, resulting in a +76% weight over-estimate. Conversely, `--dtype float32` doubles weight memory; the planner reads the model's native BF16 dtype and under-estimates by -50%. Both are unsupported inputs today.

For the complete per-model and per-configuration breakdown (TP, PP, quantization, and context length sensitivity tables), see the [full accuracy report](https://github.com/llm-d-incubation/llm-d-planner/blob/main/accuracy/accuracy_report.md).

---

## Join the Community

We covered 35 architectures. The LLM landscape ships new ones every week, and vLLM keeps evolving. The sweep runner in `accuracy/` is self-contained; run it against your cluster, submit a PR, and everyone gets the improvement.

- [GitHub: llm-d-incubation/llm-d-planner](https://github.com/llm-d-incubation/llm-d-planner)
- [Full accuracy report with per-model tables](https://github.com/llm-d-incubation/llm-d-planner/blob/main/accuracy/accuracy_report.md)
- [Run the sweep on your own cluster](https://github.com/llm-d-incubation/llm-d-planner/blob/main/accuracy/README.md)

No one should have to guess how many GPUs they need.
