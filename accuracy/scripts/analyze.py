#!/usr/bin/env python3
"""
Analyze capacity planner predictions vs actual vLLM measurements.
Generates a markdown report with error statistics per memory component.
"""

import csv
import math
import statistics
from pathlib import Path

REPO = Path(__file__).parent.parent
RAW_CSV = REPO / "results/v0.19.0/results_raw.csv"
PRED_CSV = REPO / "results/v0.19.0/results_predicted.csv"
OUT_MD = REPO / "results/v0.19.0/accuracy_report.md"


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_error(actual: float, predicted: float) -> float:
    """(predicted - actual) / actual * 100. Positive = over-estimate."""
    if actual == 0:
        return float("nan")
    return (predicted - actual) / actual * 100.0


def fmt(v: float, decimals: int = 2) -> str:
    if math.isnan(v):
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def stats(values: list[float]) -> dict:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "abs_mean": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "abs_mean": statistics.mean(abs(v) for v in vals),
        "n": len(vals),
    }


def stats_row(label: str, errors: list[float]) -> str:
    s = stats(errors)
    if s["n"] == 0:
        return f"| {label} | — | — | — | — | — | — |"
    return (
        f"| {label} | {fmt(s['mean'])} | {fmt(s['median'])} | "
        f"{fmt(s['abs_mean'])} | {fmt(s['min'])} | {fmt(s['max'])} | {s['n']} |"
    )


def fv(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}" if not math.isnan(v) else "n/a"


# ── Data Loading ──────────────────────────────────────────────────────────────

raw_ok   = [r for r in csv.DictReader(RAW_CSV.open()) if r["status"] == "ok"]
pred_all = list(csv.DictReader(PRED_CSV.open()))

def _row_key(r: dict) -> tuple:
    return (r["model"], r["tp"], r["pp"], r["dp"], r["max_model_len"],
            r["dtype"], r.get("quantization", ""), r.get("kv_cache_dtype", "auto"))

pred_map: dict[tuple, list] = {}
for p in pred_all:
    pred_map.setdefault(_row_key(p), []).append(p)

# Consume predictions in order; each raw row pops one matching prediction.
pairs: list[tuple] = []
_counts: dict[tuple, int] = {}
for raw in raw_ok:
    k = _row_key(raw)
    bucket = pred_map.get(k, [])
    idx = _counts.get(k, 0)
    if idx < len(bucket):
        pairs.append((raw, bucket[idx]))
        _counts[k] = idx + 1
    # rows with no matching prediction are silently skipped

# ── Per-row error calculation ─────────────────────────────────────────────────

COMPONENTS = {
    "weight":        ("weight_memory_gib",      "pred_weight_memory_gib"),
    "activation":    ("activation_memory_gib",  "pred_activation_memory_gib"),
    "non_torch":     ("non_torch_forward_gib",  "pred_non_torch_gib"),
    "cuda_graph":    ("cuda_graph_actual_gib",  "pred_cuda_graph_gib"),
    "total_non_kv":  ("total_non_kv_cache_gib", "pred_total_non_kv_cache_gib"),
    "kv_cache":      ("kv_cache_memory_gib",    "pred_kv_cache_memory_gib"),
    "kv_tokens":     ("kv_cache_tokens",         "pred_kv_cache_tokens"),
    "max_concurrency": ("max_concurrency",       "pred_max_concurrency"),
}

rows_data = []
for raw, pred in pairs:
    entry = {
        "log_file":      raw["log_file"],
        "model":         raw["model"],
        "architecture":  pred["architecture"],
        "gpu":           raw["gpu"],
        "tp":            int(raw["tp"]),
        "pp":            int(raw["pp"]),
        "dp":            int(raw["dp"]),
        "max_model_len": int(raw["max_model_len"]),
        "quantization":  raw["quantization"],
        "kv_cache_dtype":raw["kv_cache_dtype"],
        "dtype":         raw["dtype"],
    }
    for key, (rcol, pcol) in COMPONENTS.items():
        try:
            a = float(raw.get(rcol, ""))
            p = float(pred.get(pcol, ""))
        except (ValueError, TypeError):
            a = p = float("nan")
        entry[f"actual_{key}"] = a
        entry[f"pred_{key}"]   = p
        entry[f"err_{key}"]    = pct_error(a, p)
    rows_data.append(entry)


# ── Segment helpers ───────────────────────────────────────────────────────────

def where(fn):
    return [r for r in rows_data if fn(r)]

base  = where(lambda r: r["tp"] == 1 and r["pp"] == 1 and r["max_model_len"] == 8192
              and r["quantization"] in ("None", "", None) and r["kv_cache_dtype"] != "fp8")
multi = where(lambda r: r["tp"] > 1 or r["pp"] > 1)
quant = where(lambda r: r["quantization"] not in ("None", "", None))
kvfp8 = where(lambda r: r["kv_cache_dtype"] == "fp8")


# ── Report builder ────────────────────────────────────────────────────────────

lines = []
W = lines.append


def section(title: str, rows: list[dict]):
    W(f"\n### {title}  (n={len(rows)})\n")
    W("| Component | Mean error | Median | Mean abs | Min | Max | n |")
    W("|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|")
    for key in ["weight", "activation", "non_torch", "cuda_graph",
                "total_non_kv", "kv_cache", "max_concurrency"]:
        W(stats_row(key.replace("_", " ").title(), [r[f"err_{key}"] for r in rows]))


# ═══════════════════════════════════════════════════════════════════════════════
W("# Capacity Planner Accuracy Report — vLLM v0.19.0 / H100-80GB")
W("")
W(f"**Dataset**: {len(rows_data)} successful runs across "
  f"{len(set(r['model'] for r in rows_data))} unique models  ")
W("**Hardware**: H100-80GB (catalog memory = 80 GiB, actual = ~79.19 GiB)  ")
W("**Planner GPU util**: actual `gpu_memory_utilization` per run (0.95)  ")
W("")

# ── Executive Summary ─────────────────────────────────────────────────────────
W("## Executive Summary\n")

kv_errs_all  = [r["err_kv_cache"]      for r in rows_data if not math.isnan(r["err_kv_cache"])]
kv_errs_base = [r["err_kv_cache"]      for r in base      if not math.isnan(r["err_kv_cache"])]
act_errs     = [r["err_activation"]    for r in rows_data if not math.isnan(r["err_activation"])]
wt_errs      = [r["err_weight"]        for r in rows_data if not math.isnan(r["err_weight"])]
nt_errs      = [r["err_non_torch"]     for r in rows_data if not math.isnan(r["err_non_torch"])]
conc_errs    = [r["err_max_concurrency"] for r in rows_data if not math.isnan(r["err_max_concurrency"])]

kv_mean_all  = statistics.mean(kv_errs_all)
kv_abs_all   = statistics.mean(abs(e) for e in kv_errs_all)
kv_mean_base = statistics.mean(kv_errs_base)
act_mean     = statistics.mean(act_errs)
act_abs      = statistics.mean(abs(e) for e in act_errs)
wt_mean      = statistics.mean(wt_errs)
wt_abs       = statistics.mean(abs(e) for e in wt_errs)
conc_mean    = statistics.mean(conc_errs)
conc_abs     = statistics.mean(abs(e) for e in conc_errs)

W("| Metric | Mean error | Mean abs error | Notes |")
W("|--------|:----------:|:--------------:|-------|")
W(f"| **KV Cache memory** (all 47 runs) | {fmt(kv_mean_all)} | {fmt(kv_abs_all)} | |")
W(f"| **KV Cache memory** (baseline: tp=pp=1, len=8192, no-quant) | {fmt(kv_mean_base)} | — | n={len(kv_errs_base)} |")
W(f"| **Weight memory** | {fmt(wt_mean)} | {fmt(wt_abs)} | From safetensors metadata |")
W(f"| **Activation memory** | {fmt(act_mean)} | {fmt(act_abs)} | Largest error source |")
W(f"| **Non-torch overhead** | {fmt(statistics.mean(nt_errs))} | {fmt(statistics.mean(abs(e) for e in nt_errs))} | |")
W(f"| **Max concurrency** | {fmt(conc_mean)} | {fmt(conc_abs)} | Proxy for KV cache accuracy |")
W("")
W("### Key Findings\n")
W(f"1. **Weights are accurate** — mean abs error {fmt(wt_abs)}, computed directly from "
  "safetensors parameter counts. Errors arise only when `--dtype` overrides the native "
  "dtype (e.g., `--dtype float32`) or when quantization is not fully captured in the config.")
W(f"2. **Activation is the dominant error source** — mean {fmt(act_mean)} (over-estimate). "
  "The planner uses empirical constants (4.8–8.0 GiB) measured at `max_model_len=16000`; "
  "vLLM v0.19.0 reports 0.75–2.2 GiB across all architectures tested. Granite is worst (+600%), "
  "Mistral3/Pixtral is best (+15–23%).")
W("3. **Over-estimated activation partially cancels** the catalog GPU memory inflation (+0.77 GiB), "
  f"leaving KV cache only {fmt(kv_mean_all)} off on average across all runs. But this is "
  "coincidental cancellation of two large opposing errors, not model accuracy.")
W("4. **Non-default KV dtype (`--kv-cache-dtype fp8`) doubles token capacity** but the planner "
  "ignores this flag — KV token count is off by ~2× for those runs.")
W("5. **`--dtype float32` breaks weight prediction** — the planner uses the HuggingFace "
  "config dtype (BF16) and never sees the vLLM `--dtype` override, giving −50% weight error.")
W("6. **Pipeline parallelism reduces actual activation** (each GPU processes fewer layers) "
  "but the formula uses the same constant regardless of PP, compounding the activation error.")
W("")

# ── Component Error Tables ────────────────────────────────────────────────────
W("## Component-Level Error Breakdown\n")
W("> Percent error = (predicted − actual) / actual × 100. "
  "Positive = over-estimate, negative = under-estimate.\n")

section("All 47 Runs", rows_data)
section("Baseline: TP=1, PP=1, len=8192, no quantization, default KV dtype", base)
section("Multi-GPU (TP > 1 or PP > 1)", multi)
section("Quantized Models (fp8-dynamic / w8a8 / w4a16)", quant)
section("Non-default KV cache dtype (--kv-cache-dtype fp8)", kvfp8)

# ── Per-Model Error Table ─────────────────────────────────────────────────────
W("\n## Per-Model Errors — Baseline Runs\n")
W("> TP=1, PP=1, max_model_len=8192, no quantization, default KV dtype.\n")
W("| Model | Arch | Weight err | Activation err | Non-torch err | KV cache err | Max conc err |")
W("|-------|------|:----------:|:--------------:|:-------------:|:------------:|:------------:|")
for r in sorted(base, key=lambda x: x["model"]):
    model_short = r["model"].split("/")[-1][:35]
    arch_short  = (r["architecture"]
                   .replace("ForCausalLM", "")
                   .replace("ForConditionalGeneration", "*"))[:25]
    W(f"| {model_short} | {arch_short} | "
      f"{fmt(r['err_weight'])} | {fmt(r['err_activation'])} | "
      f"{fmt(r['err_non_torch'])} | {fmt(r['err_kv_cache'])} | "
      f"{fmt(r['err_max_concurrency'])} |")

# ═══════════════════════════════════════════════════════════════════════════════
W("\n## Argument Sensitivity Analysis\n")
W("> This section examines how each vLLM launch argument affects whether the "
  "capacity planner's memory predictions remain accurate.\n")

# ── max_model_len ─────────────────────────────────────────────────────────────
W("### `--max-model-len` (context window size)\n")

llama_len = where(lambda r: "Llama-3.1-8B-Instruct" in r["model"]
                  and r["tp"] == 1 and r["pp"] == 1
                  and r["quantization"] in ("None", "", None)
                  and r["kv_cache_dtype"] != "fp8")
llama_len.sort(key=lambda r: r["max_model_len"])

qwen_len = where(lambda r: r["model"] == "Qwen/Qwen2.5-7B-Instruct"
                 and r["tp"] == 1 and r["pp"] == 1
                 and r["quantization"] in ("None", "", None)
                 and r["kv_cache_dtype"] != "fp8")
qwen_len.sort(key=lambda r: r["max_model_len"])

W("| Model | max_model_len | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err | Max conc err |")
W("|-------|:-------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|:------------:|")
for r in llama_len + qwen_len:
    model_short = r["model"].split("/")[-1][:28]
    W(f"| {model_short} | {r['max_model_len']:,} | "
      f"{fv(r['actual_kv_cache'])} | {fmt(r['err_kv_cache'])} | "
      f"{int(r['actual_kv_tokens']):,} | {int(r['pred_kv_tokens']):,} | "
      f"{fmt(r['err_kv_tokens'])} | {fmt(r['err_max_concurrency'])} |")

W("")
W("**Conclusion**: `--max-model-len` has **no effect on KV pool size** — the formula and "
  "vLLM agree on this. Activation memory is constant (the fixed profiling overhead does not "
  "depend on context length), so the KV pool prediction error stays flat at ~−3 to −4% "
  "regardless of whether context is 2 K or 32 K tokens. The token/concurrency predictions "
  "carry that same constant KV error forward, plus any error from the per-token KV formula.")

# ── TP ────────────────────────────────────────────────────────────────────────
W("\n### `--tensor-parallel-size` (TP)\n")

tp_sweep = where(lambda r: r["model"] == "meta-llama/Llama-3.1-8B-Instruct"
                 and r["max_model_len"] == 8192
                 and r["quantization"] in ("None", "", None)
                 and r["kv_cache_dtype"] != "fp8"
                 and r["pp"] == 1)
tp_sweep.sort(key=lambda r: r["tp"])

qwen_tp = where(lambda r: r["model"] == "Qwen/Qwen2.5-7B-Instruct"
                and r["max_model_len"] == 8192
                and r["quantization"] in ("None", "", None)
                and r["kv_cache_dtype"] != "fp8"
                and r["pp"] == 1)
qwen_tp.sort(key=lambda r: r["tp"])

W("| Model | TP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |")
W("|-------|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|")
for r in tp_sweep + qwen_tp:
    model_short = r["model"].split("/")[-1][:22]
    W(f"| {model_short} | {r['tp']} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_activation'])} | {fmt(r['err_activation'])} | "
      f"{fv(r['actual_non_torch'])} | {fmt(r['err_non_torch'])} | "
      f"{fmt(r['err_kv_cache'])} |")

W("")
W("**Conclusions**:\n")
W("- **Weights scale correctly**: the formula divides by TP, matching vLLM's per-GPU weight sharding. "
  "Weight error stays near 0% across TP=1–4.")
W("- **Activation is TP-invariant in both formula and reality**: vLLM's profiling overhead does not "
  "shrink with TP (it captures the same set of batch sizes). The formula also keeps activation "
  "constant with TP. Error stays flat.")
W("- **Non-torch is heavily under-estimated for TP≥2**: the 0.60 GiB/GPU constant does not capture "
  "NCCL all-reduce buffer overhead, which grows with TP. Actual non-torch reaches ~2.1 GiB/GPU at "
  "TP=4 (3.5× the constant). However, this error is partially masked in KV cache accuracy because "
  "the over-estimated activation pulls the prediction in the opposite direction.")

# ── PP ────────────────────────────────────────────────────────────────────────
W("\n### `--pipeline-parallel-size` (PP)\n")

pp_sweep = where(lambda r: r["model"] == "meta-llama/Llama-3.1-8B-Instruct"
                 and r["max_model_len"] == 8192
                 and r["quantization"] in ("None", "", None)
                 and r["kv_cache_dtype"] != "fp8"
                 and r["tp"] == 1)
pp_sweep.sort(key=lambda r: r["pp"])

W("| PP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |")
W("|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|")
for r in pp_sweep:
    W(f"| {r['pp']} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_activation'])} | {fmt(r['err_activation'])} | "
      f"{fv(r['actual_non_torch'])} | {fmt(r['err_non_torch'])} | "
      f"{fmt(r['err_kv_cache'])} |")

# Compute activation values directly for the prose
pp_acts  = {r["pp"]: r["actual_activation"] for r in pp_sweep}
pp_preds = {r["pp"]: r["pred_activation"]   for r in pp_sweep}
W("")
W("**Conclusions**:\n")
W(f"- **Activation drops sharply with PP**: at PP=1, vLLM profiles {fv(pp_acts.get(1,float('nan')))} GiB "
  f"of activation; at PP=2 it drops to {fv(pp_acts.get(2,float('nan')))} GiB; "
  f"at PP=4 to {fv(pp_acts.get(4,float('nan')))} GiB. "
  "Each pipeline stage runs fewer transformer layers, so the profiling sweep allocates proportionally less. "
  f"The formula does not account for this and always predicts {fv(pp_preds.get(1,float('nan')))} GiB, "
  "making the activation error grow with PP (from ~+154% at PP=1 to ~+357% at PP=4).")
W("- **Non-torch increases with PP** due to inter-stage P2P send/receive buffers, "
  "but the formula uses the same TP=1 constant (0.15 GiB/GPU) regardless of PP, "
  "causing the non-torch estimate to overshoot actual (predicted > actual for PP>1 because "
  "each stage is a separate process and 0.15 is per-GPU). "
  "These two errors partially offset each other in the KV cache prediction.")
W("- **Weight error grows with PP**: the formula divides only by TP×PP for weight sharding, "
  "but with PP=4, model layers are not uniformly distributed across stages in all cases "
  "(irregular last-stage allocation can leave a stage with fewer params).")

# ── dtype ─────────────────────────────────────────────────────────────────────
W("\n### `--dtype` (compute/storage dtype override)\n")

dtype_sweep = where(lambda r: "Llama-3.1" in r["model"]
                   and r["tp"] == 1 and r["pp"] == 1 and r["max_model_len"] == 8192)
dtype_sweep.sort(key=lambda r: (r["dtype"], r["quantization"], r["kv_cache_dtype"]))

W("| dtype arg | quantization | kv_cache_dtype | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |")
W("|-----------|:------------:|:--------------:|:-------------------:|:----------:|:---------------:|:------:|")
for r in dtype_sweep:
    W(f"| {r['dtype'].replace('torch.', '')} | {r['quantization']} | "
      f"{r['kv_cache_dtype']} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_kv_cache'])} | {fmt(r['err_kv_cache'])} |")

W("")
W("**Conclusions**:\n")
W("- **`--dtype float32`** doubles model weight memory (29.98 GiB vs BF16's 14.99 GiB). "
  "The planner reads the HuggingFace config dtype (BF16) and is unaware of the `--dtype` "
  "vLLM override, so it predicts 14.96 GiB — a **−50% weight error**, which cascades into "
  "a +31% KV cache over-prediction (the planner thinks there is more room than there is).")
W("- **`--dtype float16`** is handled correctly because the HuggingFace config also stores "
  "float16 for these models; weight error stays near 0%.")
W("- **FP8-dynamic quantization** (`fp8` in the quantization column) halves weight memory. "
  "The planner reads `quantization_config` from the HuggingFace repo and applies the FP8 "
  "byte-per-param, yielding near-zero weight error. KV cache error stays consistent with "
  "the activation over-estimation.")
W("- **`--kv-cache-dtype fp8`** does not affect weight or activation predictions, but halves "
  "per-token KV storage. The planner ignores this flag and predicts KV tokens ~50% too low "
  "(see dedicated section below).")

# ── quantization ──────────────────────────────────────────────────────────────
W("\n### `--quantization` (weight quantization method)\n")

quant_rows = where(lambda r: r["quantization"] not in ("None", "", None))
quant_rows.sort(key=lambda r: (r["quantization"], r["model"]))

W("| Model | quant method | TP | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |")
W("|-------|--------------|----|:-------------------:|:----------:|:---------------:|:------:|")
for r in quant_rows:
    model_short = r["model"].split("/")[-1][:30]
    W(f"| {model_short} | {r['quantization']} | {r['tp']} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_kv_cache'])} | {fmt(r['err_kv_cache'])} |")

W("")
W("**Conclusions**:\n")
W("- **w8a8 (compressed-tensors INT8)**: the planner parses `config_groups` from the "
  "`quantization_config` to find `num_bits=8` and applies 1 byte/param. Weight errors "
  "are near zero (−0.3 to −0.7%), indicating the INT8 parameter count is well-captured.")
W("- **w4a16 (GPTQ-marlin INT4)**: the planner parses `num_bits=4` from the quantization "
  "config and applies 0.5 bytes/param. Weight error is small (~−0.7%). "
  "The large reduction in weights (5.3 GiB vs 15 GiB for BF16) frees more KV cache, "
  "and the planner correctly tracks this effect — KV error stays in the −3% range.")
W("- **fp8-dynamic** (fp8 per-tensor dynamic quant via `compressed-tensors`): "
  "the planner extracts fp8 precision from the quantization config. "
  "Weight error is near zero. Unexpectedly, weight error for the RedHat fp8 70B model "
  "at TP=2 stays very low, confirming the quant config parsing is correct for this variant.")

# ── kv_cache_dtype ────────────────────────────────────────────────────────────
W("\n### `--kv-cache-dtype` (KV cache precision)\n")

kv_dtype_rows = where(lambda r: r["kv_cache_dtype"] == "fp8")
kv_dtype_rows.sort(key=lambda r: r["model"])

# Find the matching default-kv rows for the same model
kv_default_rows = []
for kfp8 in kv_dtype_rows:
    match = where(lambda r, m=kfp8: (r["model"] == m["model"]
                                     and r["tp"] == m["tp"]
                                     and r["pp"] == m["pp"]
                                     and r["max_model_len"] == m["max_model_len"]
                                     and r["kv_cache_dtype"] != "fp8"
                                     and r["quantization"] in ("None","",None)))
    if match:
        kv_default_rows.append(match[0])

W("| Model | kv_cache_dtype | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err | Conc err |")
W("|-------|:--------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|:--------:|")
for row_pair in zip(kv_default_rows, kv_dtype_rows):
    for r in row_pair:
        model_short = r["model"].split("/")[-1][:28]
        W(f"| {model_short} | {r['kv_cache_dtype']} | "
          f"{fv(r['actual_kv_cache'])} | {fmt(r['err_kv_cache'])} | "
          f"{int(r['actual_kv_tokens']):,} | {int(r['pred_kv_tokens']):,} | "
          f"{fmt(r['err_kv_tokens'])} | {fmt(r['err_max_concurrency'])} |")
    W("|||||||||")

W("")
W("**Conclusion**: `--kv-cache-dtype fp8` stores each KV element in 1 byte instead of 2 bytes "
  "(BF16/FP16), doubling the number of tokens that fit in the KV pool. The KV pool size in GiB "
  "is unaffected (same activation and weight overhead), so the **KV GiB error stays near −4%** "
  "(the same as the default-dtype baseline). But because the planner always computes per-token "
  "bytes from the model's native compute dtype, **token count and max-concurrency predictions "
  "are ~52% too low** for fp8-KV runs. This is a direct, fixable bug: the planner should accept "
  "`kv_cache_dtype` as an input parameter and apply 1 byte/token when it is `fp8`.")

# ── Root Cause Summary ────────────────────────────────────────────────────────
W("\n## Root Cause Analysis\n")

W("### 1. Activation Memory — Largest Error Source\n")
W("The planner uses **fixed constants per architecture** (e.g., 4.8 GiB for Llama, "
  "5.6 GiB for Qwen2/3) empirically measured at `max_model_len=16000`. "
  "vLLM v0.19.0 reports substantially lower values during its profiling phase:\n")
W("| Architecture | Predicted (GiB) | Observed range (GiB) | Error range |")
W("|-------------|:---------------:|:--------------------:|:-----------:|")
archs_seen: dict[str, list] = {}
for r in rows_data:
    arch = r["architecture"]
    if not math.isnan(r["err_activation"]):
        archs_seen.setdefault(arch, []).append(
            (r["actual_activation"], r["pred_activation"], r["err_activation"]))
for arch, data in sorted(archs_seen.items()):
    acts  = [d[0] for d in data]
    preds = [d[1] for d in data]
    errs  = [d[2] for d in data]
    arch_label = (arch.replace("ForCausalLM", "")
                      .replace("ForConditionalGeneration", "*"))[:35]
    W(f"| {arch_label} | {fv(statistics.mean(preds))} | "
      f"{fv(min(acts))}–{fv(max(acts))} | "
      f"{fmt(min(errs))} to {fmt(max(errs))} |")
W("")
W("The discrepancy suggests the constants were measured with an older vLLM version or "
  "different compilation settings. Re-calibrating to these v0.19.0 measurements would be "
  "the highest-value fix.")

W("\n### 2. Non-torch Memory — Underestimated for Multi-GPU\n")
W("| TP | PP | Constant used | Actual mean (GiB) | Mean error |")
W("|:--:|:--:|:-------------:|:-----------------:|:----------:|")
for tp_v, pp_v in [(1,1),(1,2),(1,4),(2,1),(4,1)]:
    grp = where(lambda r, t=tp_v, p=pp_v: r["tp"]==t and r["pp"]==p)
    if not grp:
        continue
    const = 0.15 if tp_v == 1 else 0.60
    acts  = [r["actual_non_torch"] for r in grp if not math.isnan(r["actual_non_torch"])]
    errs  = [r["err_non_torch"]    for r in grp if not math.isnan(r["err_non_torch"])]
    if not acts:
        continue
    W(f"| {tp_v} | {pp_v} | {const} GiB | {fv(statistics.mean(acts))} | "
      f"{fmt(statistics.mean(errs))} |")

W("\nFor TP=1 the formula slightly under-estimates (0.15 vs ~0.25 GiB actual). "
  "For TP≥2, NCCL all-reduce buffers push actual non-torch to ~2.1 GiB — 3.5× "
  "the 0.60 GiB constant. For PP≥2, P2P send/receive adds overhead that the formula "
  "doesn't model at all.")

W("\n### 3. GPU Memory Catalog vs Physical\n")
W("The planner uses 80 GiB (catalog) but H100 physical VRAM is 79.19 GiB:\n")
W("- Catalog available: 80 × 0.95 = **76.00 GiB**")
W("- Physical available: 79.19 × 0.95 = **75.23 GiB**")
W("- Systematic KV over-prediction from this source alone: **+0.77 GiB**")

W("\n### 4. CUDA Graph Memory — Excluded from Formula\n")
cg_vals = [r["actual_cuda_graph"] for r in rows_data
           if not math.isnan(r["actual_cuda_graph"]) and r["actual_cuda_graph"] > 0]
W("The planner returns 0.0 GiB for CUDA graphs (treating it as included in activation). "
  "vLLM allocates the CUDA graph pool *after* sizing the KV cache, so the reported "
  "KV pool includes CUDA graph memory. The formula is therefore consistent with the "
  "log-reported KV number — no fix needed, but it should be documented.")
if cg_vals:
    W(f"\nObserved CUDA graph pool sizes: {fv(min(cg_vals))}–{fv(max(cg_vals))} GiB "
      f"(mean {fv(statistics.mean(cg_vals))} GiB).")

# ── Recommendations ───────────────────────────────────────────────────────────
W("\n## Recommendations\n")
W("| Priority | Fix | Expected impact |")
W("|:--------:|-----|:---------------:|")
W("| 🔴 High | **Re-calibrate activation constants** from v0.19.0 measurements. "
  "Current constants are 2–7× too high. Updating to ~1.0–2.2 GiB/architecture would "
  "remove the single largest prediction error. | +4–10 GiB KV accuracy |")
W("| 🔴 High | **Accept `--kv-cache-dtype` as a planner input.** When set to `fp8`, "
  "halve the per-token KV bytes. This is a one-line formula change. "
  "| 2× token/concurrency accuracy for fp8-KV runs |")
W("| 🔴 High | **Accept `--dtype` as a planner input.** When set to `float32`, "
  "double the per-param bytes for weight estimation. "
  "| Fixes −50% weight error for float32 runs |")
W("| 🟡 Medium | **Re-measure non-torch constants for TP≥2 and PP≥2.** "
  "NCCL overhead scales with both and is currently under-estimated by ~3.5×. "
  "| +1–2 GiB KV accuracy for multi-GPU |")
W("| 🟡 Medium | **Scale activation constant by 1/PP.** "
  "Each pipeline stage processes layers/PP transformer blocks; "
  "profiling overhead scales proportionally. "
  "| Fixes growing activation error at high PP |")
W("| 🟢 Low | **Use physical GPU memory** (79.19 GiB for H100) rather than "
  "the catalog 80 GiB nominal. | +0.77 GiB KV accuracy |")

report = "\n".join(lines)
OUT_MD.write_text(report)
print(f"Report written → {OUT_MD}")
print(f"\n{'─'*60}")
print("HEADLINE NUMBERS")
print(f"{'─'*60}")
print(f"  KV cache mean error (all):      {fmt(kv_mean_all)}")
print(f"  KV cache mean error (baseline): {fmt(kv_mean_base)}")
print(f"  Weights mean abs error:         {fmt(wt_abs)}")
print(f"  Activation mean error:          {fmt(act_mean)}")
print(f"  Activation mean abs error:      {fmt(act_abs)}")
print(f"  Max concurrency mean error:     {fmt(conc_mean)}")
print(f"  Max concurrency mean abs error: {fmt(conc_abs)}")
