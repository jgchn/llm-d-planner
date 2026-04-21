"""
Aggregate per-run JSON files → CSV + Markdown report.

Supports two input formats:
  1. Flat vLLM-log format (from parse_log.py / sweep_runner.py):
       {"model": "...", "gpu": "H100-80GB", "vllm_args": {...},
        "weight_memory_gib": 14.99, "kv_cache_memory_gib": 58.11, ...}
     analyze.py fetches model configs from HuggingFace and calls the
     capacity planner to compute predictions automatically.

  2. Pre-analyzed format (legacy):
       {"model": "...", "tp": 1, ...,
        "measured": {"weight_memory_gib": ..., "kv_cache_gib": ..., ...},
        "planner_predicted": {...}}

Usage:
    python analyze.py --runs accuracy/results/v0.19.0/runs/ \
                      --out  accuracy/results/v0.19.0/report.md \
                      --csv  accuracy/results/v0.19.0/results.csv \
                      [--hf-token <token>]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

COMPONENTS = ["weight_memory", "activation_memory", "non_torch_memory", "kv_cache"]
_MKEYS = {
    "weight_memory":     "weight_memory_gib",
    "activation_memory": "activation_memory_gib",
    "non_torch_memory":  "non_torch_memory_gib",
    "kv_cache":          "kv_cache_gib",
}

# Known GPU memory sizes (GiB). _gpu_memory_gib() falls back to regex parsing.
_GPU_MEMORY_GIB: dict[str, int] = {
    "H100-80GB": 80, "H100-40GB": 40,
    "A100-80GB": 80, "A100-40GB": 40,
    "L40S": 48, "L4": 24, "A10G": 24, "A10": 24,
    "V100-32GB": 32, "V100-16GB": 16,
}


def _gpu_memory_gib(gpu_name: str) -> int:
    if gpu_name in _GPU_MEMORY_GIB:
        return _GPU_MEMORY_GIB[gpu_name]
    m = re.search(r"(\d+)\s*GB", gpu_name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine GPU memory for: {gpu_name!r}")


def compute_planner_predictions(run: dict[str, Any], hf_token: str | None = None) -> dict[str, float]:
    """Call the capacity planner for the given run's model + vllm_args."""
    from planner.capacity_planner import (
        allocatable_kv_cache_memory,
        estimate_vllm_activation_memory,
        estimate_vllm_cuda_graph_memory,
        estimate_vllm_non_torch_memory,
        get_model_config_from_hf,
        per_gpu_model_memory_required,
    )

    model_name: str = run["model"]
    va: dict = run.get("vllm_args", run)
    tp = int(va.get("tensor_parallel_size", run.get("tp", 1)))
    pp = int(va.get("pipeline_parallel_size", run.get("pp", 1)))
    dp = int(va.get("data_parallel_size", run.get("dp", 1)))
    max_model_len = int(va.get("max_model_len", run.get("max_model_len", 8192)))
    gpu_util = float(va.get("gpu_memory_utilization", 0.9))
    gpu_memory = _gpu_memory_gib(run["gpu"])

    model_config = get_model_config_from_hf(model_name, hf_token)
    weight = per_gpu_model_memory_required(model_name, model_config, tp, pp, hf_token)
    kv = allocatable_kv_cache_memory(
        model_name, model_config, gpu_memory, gpu_util,
        tp=tp, pp=pp, dp=dp, max_model_len=max_model_len, hf_token=hf_token,
    )
    activation = estimate_vllm_activation_memory(model_config, tp=tp)
    non_torch = estimate_vllm_non_torch_memory(tp)
    cuda_graph = estimate_vllm_cuda_graph_memory()

    # allocatable_kv_cache_memory() returns total KV across all (tp×pp) GPUs.
    # vLLM logs "Available KV cache memory" per GPU, so divide to match.
    kv_per_gpu = kv / (tp * pp)

    return {
        "weight_memory_gib":     round(weight, 2),
        "activation_memory_gib": round(activation, 2),
        "non_torch_memory_gib":  round(non_torch, 2),
        "kv_cache_gib":          round(kv_per_gpu, 2),
        "kv_cache_total_gib":    round(kv, 2),
        "cuda_graph_memory_gib": round(cuda_graph, 2),
    }


def _normalize_run(run: dict[str, Any], hf_token: str | None = None) -> dict[str, Any]:
    """Convert flat vLLM-log format to analyzed format; no-op for pre-analyzed format."""
    if "measured" in run or "planner_predicted" in run:
        return run

    va: dict = run.get("vllm_args", {})
    normalized: dict[str, Any] = {
        "model":         run["model"],
        "gpu":           run.get("gpu", "unknown"),
        "tp":            int(va.get("tensor_parallel_size", 1)),
        "pp":            int(va.get("pipeline_parallel_size", 1)),
        "dp":            int(va.get("data_parallel_size", 1)),
        "max_model_len": int(va.get("max_model_len", 8192)),
        "vllm_args":     va,
    }
    if "_sweep_dim" in run:
        normalized["_sweep_dim"] = run["_sweep_dim"]

    # activation_memory and non_torch_memory are not directly logged by vLLM
    normalized["measured"] = {
        "weight_memory_gib":     run.get("weight_memory_gib"),
        "kv_cache_gib":          run.get("kv_cache_memory_gib"),
        "activation_memory_gib": None,
        "non_torch_memory_gib":  None,
    }

    try:
        normalized["planner_predicted"] = compute_planner_predictions(run, hf_token)
    except Exception as exc:
        print(
            f"Warning: planner prediction failed for {run['model']}: {exc}",
            file=sys.stderr,
        )

    return normalized


def compute_error_pct(run: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for c in COMPONENTS:
        key = _MKEYS[c]
        measured = run["measured"].get(key)
        predicted = run["planner_predicted"].get(key)
        if measured is not None and predicted is not None and measured != 0:
            result[c] = (predicted - measured) / measured * 100
        else:
            result[c] = None
    return result


def load_runs(directory: str | Path, hf_token: str | None = None) -> list[dict[str, Any]]:
    runs = []
    for p in sorted(Path(directory).glob("*.json")):
        data = json.loads(p.read_text())
        if data.get("skipped"):
            continue
        data = _normalize_run(data, hf_token)
        if "error_pct" not in data and "planner_predicted" in data:
            data["error_pct"] = compute_error_pct(data)
        runs.append(data)
    return runs


def find_outliers(runs: list[dict[str, Any]], threshold_pct: float = 10.0) -> list[dict[str, Any]]:
    return [
        r for r in runs
        if any(
            v is not None and abs(v) > threshold_pct
            for v in r.get("error_pct", {}).values()
        )
    ]


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{'+'if v>0 else ''}{v:.1f}%"


def generate_markdown_report(runs: list[dict[str, Any]]) -> str:
    lines = ["# Memory Validation Report\n"]

    lines += ["## Per-component error\n",
              "| Model | TP | PP | DP | max_len | Weight | Activation | Non-torch | KV cache |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        e = r.get("error_pct", {})
        lines.append(
            f"| {r['model']} | {r['tp']} | {r['pp']} | {r['dp']} | {r['max_model_len']} "
            f"| {_fmt(e.get('weight_memory'))} | {_fmt(e.get('activation_memory'))} "
            f"| {_fmt(e.get('non_torch_memory'))} | {_fmt(e.get('kv_cache'))} |"
        )
    lines.append("")

    lines += ["## Per-architecture error\n",
              "_Group by architecture class. Mean and max absolute error per component._\n"]

    lines.append("## Argument sensitivity\n")

    def _sweep_val(r: dict, dim: str) -> Any:
        if dim in r:
            return r[dim]
        return r.get("vllm_args", {}).get(dim, "?")

    for sweep_dim in ("max_model_len", "tp", "pp", "dp", "dtype", "quantization", "kv_cache_dtype"):
        sweep_runs = [r for r in runs if r.get("_sweep_dim") == sweep_dim]
        if sweep_runs:
            lines += [f"### {sweep_dim} sweep\n",
                      "| Value | Weight | Activation | Non-torch | KV cache |",
                      "|---|---|---|---|---|"]
            for r in sweep_runs:
                e = r.get("error_pct", {})
                lines.append(
                    f"| {_sweep_val(r, sweep_dim)} "
                    f"| {_fmt(e.get('weight_memory'))} | {_fmt(e.get('activation_memory'))} "
                    f"| {_fmt(e.get('non_torch_memory'))} | {_fmt(e.get('kv_cache'))} |"
                )
            lines.append("")

    lines.append("## Outliers\n")
    outliers = find_outliers(runs)
    if outliers:
        for r in outliers:
            bad = {k: v for k, v in r.get("error_pct", {}).items() if v is not None and abs(v) > 10}
            lines.append(f"- **{r['model']}** (TP={r['tp']}): {bad} — root cause required")
    else:
        lines.append("_No outliers (all components within ±10%)._")
    lines.append("")

    lines += ["## Calibration decisions\n",
              "_Document constant changes here: old value → new value, evidence._\n"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace API token (needed for gated models)")
    args = ap.parse_args()

    runs = load_runs(args.runs, hf_token=args.hf_token)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(generate_markdown_report(runs))
    print(f"Report written to {args.out} ({len(runs)} runs)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model", "gpu", "tp", "pp", "dp", "max_model_len",
                "dtype", "quantization", "kv_cache_dtype",
                "weight_error_pct", "activation_error_pct",
                "non_torch_error_pct", "kv_cache_error_pct"])
            w.writeheader()
            for r in runs:
                e = r.get("error_pct", {})
                va = r.get("vllm_args", r)
                w.writerow({
                    "model": r["model"], "gpu": r["gpu"],
                    "tp": va.get("tensor_parallel_size", r.get("tp")),
                    "pp": va.get("pipeline_parallel_size", r.get("pp")),
                    "dp": va.get("data_parallel_size", r.get("dp")),
                    "max_model_len": va.get("max_model_len", r.get("max_model_len")),
                    "dtype": va.get("dtype", "auto"),
                    "quantization": va.get("quantization"),
                    "kv_cache_dtype": va.get("kv_cache_dtype", "auto"),
                    "weight_error_pct":     e.get("weight_memory"),
                    "activation_error_pct": e.get("activation_memory"),
                    "non_torch_error_pct":  e.get("non_torch_memory"),
                    "kv_cache_error_pct":   e.get("kv_cache"),
                })


if __name__ == "__main__":
    main()
