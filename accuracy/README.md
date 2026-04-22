# Reproducing the vLLM Memory Validation Campaign

This guide is self-contained. No knowledge of this codebase is assumed.

## What this does

A single Kubernetes Job (the **orchestrator**) runs inside your cluster and drives the full
sweep. For each entry in `sweep.yaml`, it creates a sub-job that starts `vllm serve`, waits
for the startup probe to pass (vLLM is ready), fetches the full startup log, deletes the job,
and saves a JSON result. Sub-jobs run sequentially so only one GPU workload runs at a time.

## Prerequisites

| Tool | Min version | Install |
|---|---|---|
| kubectl | 1.28 | https://kubernetes.io/docs/tasks/tools/ |
| Kubernetes cluster | 1.28 | with NVIDIA GPU Operator |
| HuggingFace account | — | https://huggingface.co (gated models need approval) |

GPU requirement: at least 1× H100 80GB (or A100 80GB). 70B model runs need TP=2+ (2+ GPUs).

## Step 1: Clone

```bash
git clone https://github.com/llm-d-incubation/llm-d-planner
cd llm-d-planner
```

## Step 2: Find your cluster's GPU node label

```bash
kubectl get nodes --show-labels | grep -i gpu
```

Open `accuracy/scripts/sweep.yaml` and update `defaults.node_selector` to match.
The comment in the file explains what to change.

## Step 3: Apply infrastructure (one-time)

```bash
kubectl apply -f accuracy/k8s/namespace.yaml
kubectl apply -f accuracy/k8s/rbac.yaml
kubectl apply -f accuracy/k8s/pvc.yaml

# Create HuggingFace token Secret (replace with your token):
kubectl create secret generic hf-token \
  --from-literal=token=hf_YOUR_TOKEN_HERE \
  --namespace llmdplanner
```

## Step 4: Sync ConfigMaps from scripts

Run these from the repo root to push the latest Python scripts and sweep config into the cluster:

```bash
kubectl create configmap vllm-mem-scripts \
  --from-file=sweep_runner.py=accuracy/scripts/sweep_runner.py \
  --from-file=parse_log.py=accuracy/scripts/parse_log.py \
  --namespace llmdplanner --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap vllm-mem-sweep \
  --from-file=sweep.yaml=accuracy/scripts/sweep.yaml \
  --namespace llmdplanner --dry-run=client -o yaml | kubectl apply -f -
```

Re-run this step any time you edit `sweep.yaml` or the Python scripts.

## Step 5: Smoke test with one run

Edit `sweep.yaml` to contain only one run entry (e.g., `Llama-3.1-8B-Instruct`, TP=1),
sync the ConfigMap (Step 4), then:

```bash
kubectl apply -f accuracy/k8s/orchestrator-job.yaml
kubectl logs -f job/vllm-mem-orchestrator -n llmdplanner
```

Expected: the orchestrator log shows `Submitted`, `Waiting for pod ready`, `Pod ready`,
`Log saved`, `JSON saved`, `Sweep complete`. Takes ~5-10 minutes.

Fetch the result:
```bash
# Exec into a pod that mounts the PVC, or copy via a reader pod:
kubectl run reader --image=busybox --restart=Never --rm -it \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"vllm-mem-data"}}],"containers":[{"name":"reader","image":"busybox","volumeMounts":[{"name":"data","mountPath":"/data"}]}]}}' \
  -n llmdplanner -- ls /data/results/
```

## Step 6: Verify log patterns

The patterns in `accuracy/scripts/parse_log.py` are validated against real vLLM v0.19.0
logs. After the smoke test, confirm the JSON result has non-null `weight_memory_gib` and
`kv_cache_memory_gib`. If a future vLLM version changes log format, update the regex
patterns, re-run `uv run pytest accuracy/tests/ -q`, and re-sync ConfigMaps before proceeding.

## Step 7: Run the full sweep

Restore `sweep.yaml` to the full matrix, sync ConfigMaps, delete the previous orchestrator
Job, and resubmit:

```bash
kubectl delete job vllm-mem-orchestrator -n llmdplanner --ignore-not-found
kubectl apply -f accuracy/k8s/orchestrator-job.yaml
kubectl logs -f job/vllm-mem-orchestrator -n llmdplanner
```

Estimated time: 8–16 hours depending on model download speed and GPU availability.
The orchestrator is restartable: if it fails mid-sweep, already-saved JSON files are skipped
on resubmit.

## Step 8: Collect results and generate the report

Pull logs and JSON results from the cluster PVC to your local machine:

```bash
python accuracy/scripts/collect.py
# Results land in: data/benchmarks/memory/v0.19.0/runs/ and .../logs/
```

Copy the new run JSONs into `accuracy/results/v0.19.0/runs/`, then generate the report.
`analyze.py` calls the capacity planner directly to compute predictions — no separate
calibration step needed. For gated models pass `--hf-token <token>` (only fetches
`config.json`, not model weights):

```bash
python accuracy/scripts/analyze.py \
  --runs accuracy/results/v0.19.0/runs/ \
  --out  accuracy/results/v0.19.0/report.md \
  --csv  accuracy/results/v0.19.0/results.csv

python accuracy/scripts/deep_analysis.py \
  --csv accuracy/results/v0.19.0/results.csv \
  --out accuracy/results/v0.19.0/deep_analysis.md
```

## Reproducing from committed results (no cluster needed)

The raw run JSONs are committed in `accuracy/results/v0.19.0/runs/`. To regenerate
the report and analysis locally:

```bash
uv run python accuracy/scripts/analyze.py \
  --runs accuracy/results/v0.19.0/runs/ \
  --out  accuracy/results/v0.19.0/report.md \
  --csv  accuracy/results/v0.19.0/results.csv

uv run python accuracy/scripts/deep_analysis.py \
  --csv accuracy/results/v0.19.0/results.csv \
  --out accuracy/results/v0.19.0/deep_analysis.md
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Sub-job pod stays Pending | Wrong node_selector or insufficient GPU quota | `kubectl describe pod -n llmdplanner -l app=vllm-mem-validation` |
| Startup probe never passes | vLLM OOM or model too large for single GPU | Check pod log for `cudaMalloc failed`; increase TP |
| Parse fails: missing field | Log format differs from patterns | Review log, update patterns in `parse_log.py`, re-sync ConfigMap |
| HF 401 error in pod log | Token missing or no access to gated model | Re-create `hf-token` Secret; request model access on HuggingFace |
| Orchestrator exits early | One sub-job failed | Check `kubectl logs job/vllm-mem-orchestrator`; resubmit after fix |
