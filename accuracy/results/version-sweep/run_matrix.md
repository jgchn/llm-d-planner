# Run Matrix — vLLM Version Sensitivity / Qwen3-14B / H100-80GB

**Goal**: Track how activation memory reported by vLLM changes across releases, to identify
when planner constants became stale.

**4 successful runs, 1 failed run (first attempt only).**

Model: `Qwen/Qwen3-14B` — tp=1, pp=1, dp=1, max_model_len=8192, dtype=auto, quant=none.
All runs on a single H100-80GB at `gpu_memory_utilization=0.95`.

## Results

| vLLM version | Weight (GiB) | Activation (GiB) | Non-torch (GiB) | KV cache (GiB) | Max concurrency |
|:---:|:---:|:---:|:---:|:---:|:---:|
| v0.15.0 | 27.52 | 5.64 | 0.13 | 41.94 | 33.55 |
| v0.16.0 | 27.52 | 5.64 | 0.13 | 41.94 | 33.55 |
| **v0.17.0** | 27.52 | **2.23** | 0.13 | 45.34 | 36.27 |
| v0.18.0 | 27.52 | 2.23 | 0.25 | 45.23 | 36.18 |

**Key finding**: Activation memory dropped from 5.64 GiB to 2.23 GiB (−60%) between v0.16.0 and v0.17.0.
Weight memory is stable across all versions (as expected — model parameters don't change).
KV cache increased by ~3.4 GiB at v0.17.0+ because lower activation overhead leaves more headroom.

The planner's Qwen3 activation constant (5.60 GiB) matches v0.16.0 exactly — the constants
were calibrated against v0.16.0 or earlier.

## Failed Runs

| vLLM version | Notes |
|:---:|---|
| v0.16.0 (attempt 1) | GPU contention at startup: only 61.8/79.19 GiB free (needed 75.23 GiB). Succeeded on retry. |
