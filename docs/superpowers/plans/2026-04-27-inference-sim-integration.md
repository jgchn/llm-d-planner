# Inference-Sim Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the planner's static benchmark DB latency lookup with live inference-sim subprocess calls, adding cache affinity comparison and reliability signals to recommendation cards, plus a per-card Explore panel for scenario analysis.

**Architecture:** A new `src/planner/simulation/` package wraps the inference-sim binary via subprocess. `config_finder.py` runs candidates in parallel through this client, using simulation latency (falling back to DB). Cache affinity simulation enriches the top-3 balanced recommendations. A new `/api/v1/explore-config` endpoint powers an Explore expander per recommendation card in the Streamlit UI.

**Tech Stack:** Python 3.11+ / uv, Go (inference-sim binary in `../inference-sim/`), Streamlit, FastAPI/Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-04-27-inference-sim-integration-design.md`

---

## Chunk 1: inference-sim Go change + Makefile

### Task 1: Add `--output json` to inference-sim

**Files:**
- Modify: `../inference-sim/sim/metrics.go`
- Modify: `../inference-sim/cmd/root.go`

- [ ] **Step 1: Read the existing files**

  ```bash
  cat ../inference-sim/sim/metrics.go
  cat ../inference-sim/cmd/root.go
  ```

  Note the exact field names in `Metrics`, how `Print()` computes percentiles, and how the `--rate` flag is defined (you'll add `--output` the same way).

- [ ] **Step 2: Add `PreemptionCount` and `KVAllocationFailures` to `Metrics` and wire them up**

  These fields do not exist in the current `Metrics` struct. Add them in three places:

  **`../inference-sim/sim/metrics.go`** — append to the `Metrics` struct body:
  ```go
  PreemptionCount      int
  KVAllocationFailures int
  ```

  **`../inference-sim/sim/simulator.go` line ~280** — inside the `preempt()` function, after `sim.PreemptionHappened = true`:
  ```go
  sim.Metrics.PreemptionCount++
  ```

  **`../inference-sim/sim/simulator.go` line ~383** — inside the block that handles `AllocateKVBlocks` returning false when scheduling a new request (the `!ok` branch at line 381 that currently just does `break`):
  ```go
  if ok := sim.KVCache.AllocateKVBlocks(next, startIndex, endIndex, cachedBlocks); !ok {
      // cannot allocate enough blocks — count as failure before breaking
      sim.Metrics.KVAllocationFailures++
      break
  }
  ```

  Also update `NewMetrics()` in `../inference-sim/sim/metrics.go` (currently at line 38) to initialize the new fields. Add these two lines inside the `return &Metrics{...}` block:
  ```go
  PreemptionCount:      0,
  KVAllocationFailures: 0,
  ```

  Verify the Simulator struct has a `Metrics *Metrics` field (it does — confirm with `grep -n "Metrics" ../inference-sim/sim/simulator.go | head -5`).

- [ ] **Step 3: Add `PrintJSON()` to `Metrics`**

  Add this to `../inference-sim/sim/metrics.go`:

  ```go
  import (
      "encoding/json"
      "fmt"
      "math"
      "sort"
      "time"
  )

  type jsonMetricsOutput struct {
      TTFTp95Ms            float64 `json:"ttft_p95_ms"`
      ITLp95Ms             float64 `json:"itl_p95_ms"`
      E2Ep95Ms             float64 `json:"e2e_p95_ms"`
      KVAllocationFailures int     `json:"kv_allocation_failures"`
      TotalRequests        int     `json:"total_requests"`
      PreemptionCount      int     `json:"preemption_count"`
      ResponsesPerSec      float64 `json:"responses_per_sec"`
  }

  func mapValues(m map[string]float64) []float64 {
      out := make([]float64, 0, len(m))
      for _, v := range m {
          out = append(out, v)
      }
      return out
  }

  func ticksToMs(ticks []int64) []float64 {
      out := make([]float64, len(ticks))
      for i, t := range ticks {
          out[i] = float64(t) / 1000.0 // 1 tick = 1 microsecond
      }
      return out
  }

  func p95(values []float64) float64 {
      if len(values) == 0 {
          return 0
      }
      sorted := make([]float64, len(values))
      copy(sorted, values)
      sort.Float64s(sorted)
      idx := int(math.Ceil(0.95*float64(len(sorted)))) - 1
      if idx < 0 {
          idx = 0
      }
      return sorted[idx]
  }

  func (m *Metrics) PrintJSON(horizon int64, startTime time.Time) {
      elapsed := time.Since(startTime).Seconds()
      if elapsed <= 0 {
          elapsed = 1
      }
      out := jsonMetricsOutput{
          TTFTp95Ms:            p95(mapValues(m.RequestTTFTs)),
          ITLp95Ms:             p95(ticksToMs(m.AllITLs)),
          E2Ep95Ms:             p95(mapValues(m.RequestE2Es)),
          KVAllocationFailures: m.KVAllocationFailures,
          TotalRequests:        m.CompletedRequests,
          PreemptionCount:      m.PreemptionCount,
          ResponsesPerSec:      float64(m.CompletedRequests) / elapsed,
      }
      b, _ := json.Marshal(out)
      fmt.Println(string(b))
  }
  ```

  > Check the existing `Print()` signature — if it takes `totalBlocks int64` use that pattern consistently, but `PrintJSON` does not need it.

- [ ] **Step 4: Add `--output` flag to `runCmd` in `cmd/root.go`**

  Add a package-level variable near the other flag variables at the top of the file:
  ```go
  var outputFormat string
  ```

  In the `init()` function alongside other `runCmd.Flags()` calls:
  ```go
  runCmd.Flags().StringVar(&outputFormat, "output", "text", "Output format: text or json")
  ```

  In the `RunE` function, find line 144 which reads:
  ```go
  s.Metrics.Print(s.Horizon, totalKVBlocks, startTime)
  ```
  (`startTime` is declared at line 128: `startTime := time.Now()`)

  Replace that single line with:
  ```go
  if outputFormat == "json" {
      s.Metrics.PrintJSON(s.Horizon, startTime)
  } else {
      s.Metrics.Print(s.Horizon, totalKVBlocks, startTime)
  }
  ```

- [ ] **Step 5: Build and verify**

  ```bash
  cd ../inference-sim && go build -o simulation_worker ./...
  ```

  Expected: builds without errors.

  Run a smoke test:
  ```bash
  ./simulation_worker run \
    --model meta-llama/llama-3.1-8b-instruct \
    --hardware H100 --tp 1 \
    --rate 5 --max-prompts 100 \
    --output json
  ```

  Expected: a single line of JSON containing `ttft_p95_ms`, `itl_p95_ms`, `e2e_p95_ms`, `kv_allocation_failures`, `total_requests`, `preemption_count`, `responses_per_sec`.

  > If the model/hardware combo isn't in `coefficients.yaml`, try a different one that is.

- [ ] **Step 6: Commit (in inference-sim repo)**

  ```bash
  cd ../inference-sim
  git add sim/metrics.go cmd/root.go
  git commit -s -m "feat: add --output json flag for machine-readable metrics"
  cd -
  ```

---

### Task 2: Makefile `setup-inference-sim` target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Locate the `setup-backend` target in Makefile**

  ```bash
  grep -n "setup-backend\|setup:" Makefile | head -10
  ```

- [ ] **Step 2: Add the target**

  Add after the `setup:` umbrella target (currently at line 151 of Makefile):
  ```makefile
  INFERENCE_SIM_BIN ?= ../inference-sim/simulation_worker

  setup-inference-sim:
  	@if [ ! -f "$(INFERENCE_SIM_BIN)" ]; then \
  		echo "Building inference-sim binary..."; \
  		if ! command -v go &> /dev/null; then \
  			echo "Warning: Go toolchain not found. Simulation will use DB benchmarks only."; \
  			echo "To enable live simulation: install Go and re-run make setup-inference-sim"; \
  		else \
  			cd $$(dirname $(INFERENCE_SIM_BIN)) && go build -o simulation_worker ./...; \
  			echo "inference-sim binary built at $(INFERENCE_SIM_BIN)"; \
  		fi \
  	else \
  		echo "inference-sim binary already exists at $(INFERENCE_SIM_BIN)"; \
  	fi
  ```

  Then add `setup-inference-sim` as the last line in the `setup-backend` target body (currently at line 131–135; add non-fatal call at end):
  ```makefile
  	$(MAKE) setup-inference-sim || true
  ```

- [ ] **Step 3: Verify**

  ```bash
  make setup-inference-sim
  ```

  Expected: either "already exists" or builds successfully.

- [ ] **Step 4: Commit**

  ```bash
  git add Makefile
  git commit -s -m "chore: add setup-inference-sim Makefile target"
  ```

---

## Chunk 2: Python simulation package

### Task 3: `simulation/client.py`

**Files:**
- Create: `src/planner/simulation/__init__.py`
- Create: `src/planner/simulation/client.py`
- Create: `tests/unit/test_simulation_client.py`

- [ ] **Step 1: Write the failing tests**

  Create `tests/unit/test_simulation_client.py`:
  ```python
  import json
  import pytest
  from unittest.mock import MagicMock, patch
  from planner.simulation.client import SimulationClient, SimulationResult


  SAMPLE_JSON = json.dumps({
      "ttft_p95_ms": 26.3,
      "itl_p95_ms": 10.7,
      "e2e_p95_ms": 3856.1,
      "kv_allocation_failures": 2,
      "total_requests": 100,
      "preemption_count": 3,
      "responses_per_sec": 9.91,
  })


  @pytest.mark.unit
  def test_simulate_returns_none_when_binary_missing():
      client = SimulationClient(bin_path="/nonexistent/binary")
      result = client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 9.0)
      assert result is None


  @pytest.mark.unit
  def test_simulate_returns_none_when_qps_zero():
      client = SimulationClient(bin_path="/nonexistent/binary")
      result = client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 0.0)
      assert result is None


  @pytest.mark.unit
  def test_simulate_returns_none_when_qps_negative():
      client = SimulationClient(bin_path="/nonexistent/binary")
      result = client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, -1.0)
      assert result is None


  @pytest.mark.unit
  def test_simulate_parses_json_output():
      mock_proc = MagicMock()
      mock_proc.returncode = 0
      mock_proc.stdout = SAMPLE_JSON

      with patch("subprocess.run", return_value=mock_proc), \
           patch("os.path.isfile", return_value=True), \
           patch("os.access", return_value=True):
          client = SimulationClient(bin_path="/fake/binary")
          result = client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 9.0)

      assert result is not None
      assert result.ttft_p95_ms == pytest.approx(26.3)
      assert result.itl_p95_ms == pytest.approx(10.7)
      assert result.e2e_p95_ms == pytest.approx(3856.1)
      assert result.kv_allocation_failure_rate == pytest.approx(2 / 100)
      assert result.preemption_rate == pytest.approx(3 / 100)
      assert result.responses_per_sec == pytest.approx(9.91)
      assert result.source == "simulation"


  @pytest.mark.unit
  def test_simulate_returns_none_on_nonzero_exit():
      mock_proc = MagicMock()
      mock_proc.returncode = 1
      mock_proc.stdout = ""

      with patch("subprocess.run", return_value=mock_proc), \
           patch("os.path.isfile", return_value=True), \
           patch("os.access", return_value=True):
          client = SimulationClient(bin_path="/fake/binary")
          result = client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 9.0)

      assert result is None


  @pytest.mark.unit
  def test_simulate_returns_none_on_invalid_json():
      mock_proc = MagicMock()
      mock_proc.returncode = 0
      mock_proc.stdout = "not json"

      with patch("subprocess.run", return_value=mock_proc), \
           patch("os.path.isfile", return_value=True), \
           patch("os.access", return_value=True):
          client = SimulationClient(bin_path="/fake/binary")
          result = client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 9.0)

      assert result is None


  @pytest.mark.unit
  def test_simulate_includes_prefix_tokens_in_command():
      mock_proc = MagicMock()
      mock_proc.returncode = 0
      mock_proc.stdout = SAMPLE_JSON

      with patch("subprocess.run", return_value=mock_proc) as mock_run, \
           patch("os.path.isfile", return_value=True), \
           patch("os.access", return_value=True):
          client = SimulationClient(bin_path="/fake/binary")
          client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 9.0, prefix_tokens=400)

      cmd = mock_run.call_args[0][0]
      assert "--prefix-tokens" in cmd
      assert "400" in cmd


  @pytest.mark.unit
  def test_simulate_omits_prefix_tokens_when_zero():
      mock_proc = MagicMock()
      mock_proc.returncode = 0
      mock_proc.stdout = SAMPLE_JSON

      with patch("subprocess.run", return_value=mock_proc) as mock_run, \
           patch("os.path.isfile", return_value=True), \
           patch("os.access", return_value=True):
          client = SimulationClient(bin_path="/fake/binary")
          client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 9.0, prefix_tokens=0)

      cmd = mock_run.call_args[0][0]
      assert "--prefix-tokens" not in cmd
  ```

- [ ] **Step 2: Run to confirm failure**

  `tests/unit/` exists in this repo (confirmed). Tests run from `src/`:
  ```bash
  cd src && uv run pytest ../tests/unit/test_simulation_client.py -v
  ```

  Expected: `ImportError` or `ModuleNotFoundError` for `planner.simulation.client`.

- [ ] **Step 3: Create `src/planner/simulation/__init__.py`**

  ```python
  from planner.simulation.client import SimulationClient, SimulationResult

  __all__ = ["SimulationClient", "SimulationResult"]
  ```

- [ ] **Step 4: Create `src/planner/simulation/client.py`**

  ```python
  import json
  import logging
  import os
  import subprocess
  from dataclasses import dataclass, field
  from typing import Literal

  logger = logging.getLogger(__name__)

  # Path breakdown: client.py is at src/planner/simulation/client.py
  # Going up 4 dirs from that file's directory lands at the workspace root,
  # where inference-sim/ is a sibling of llm-d-planner/. Override with
  # INFERENCE_SIM_BIN env var if your layout differs.
  _DEFAULT_BIN = os.path.join(
      os.path.dirname(__file__), "../../../../inference-sim/simulation_worker"
  )


  @dataclass
  class SimulationResult:
      ttft_p95_ms: float
      itl_p95_ms: float
      e2e_p95_ms: float
      kv_allocation_failure_rate: float
      preemption_rate: float
      responses_per_sec: float
      source: Literal["simulation", "database"] = "simulation"


  class SimulationClient:
      def __init__(self, bin_path: str | None = None) -> None:
          self.bin_path = bin_path or os.environ.get("INFERENCE_SIM_BIN", _DEFAULT_BIN)
          self._available: bool | None = None

      def is_available(self) -> bool:
          if self._available is None:
              self._available = os.path.isfile(self.bin_path) and os.access(
                  self.bin_path, os.X_OK
              )
          return self._available

      def simulate(
          self,
          model: str,
          gpu: str,
          tp: int,
          prompt_tokens: int,
          output_tokens: int,
          qps: float,
          prefix_tokens: int = 0,
      ) -> SimulationResult | None:
          if qps <= 0 or not self.is_available():
              return None

          cmd = [
              self.bin_path, "run",
              "--model", model,
              "--hardware", gpu,
              "--tp", str(tp),
              "--prompt-tokens", str(prompt_tokens),
              "--output-tokens", str(output_tokens),
              "--rate", str(qps),
              "--max-prompts", "500",
              "--output", "json",
          ]
          if prefix_tokens > 0:
              cmd += ["--prefix-tokens", str(prefix_tokens)]

          try:
              proc = subprocess.run(
                  cmd, capture_output=True, text=True, timeout=10
              )
              if proc.returncode != 0:
                  logger.warning("inference-sim exited %d: %s", proc.returncode, proc.stderr[:200])
                  return None
              data = json.loads(proc.stdout)
              total = data.get("total_requests") or 1
              return SimulationResult(
                  ttft_p95_ms=float(data["ttft_p95_ms"]),
                  itl_p95_ms=float(data["itl_p95_ms"]),
                  e2e_p95_ms=float(data["e2e_p95_ms"]),
                  kv_allocation_failure_rate=data["kv_allocation_failures"] / total,
                  preemption_rate=data["preemption_count"] / total,
                  responses_per_sec=float(data["responses_per_sec"]),
              )
          except subprocess.TimeoutExpired:
              logger.warning("inference-sim timed out for model=%s gpu=%s", model, gpu)
              return None
          except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
              logger.warning("inference-sim parse error: %s", e)
              return None
  ```

- [ ] **Step 5: Run tests to confirm they pass**

  ```bash
  cd src && uv run pytest ../tests/unit/test_simulation_client.py -v
  ```

  Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/planner/simulation/ tests/unit/test_simulation_client.py
  git commit -s -m "feat: add simulation client subprocess wrapper"
  ```

---

### Task 4: `simulation/cache.py`

**Files:**
- Create: `src/planner/simulation/cache.py`
- Create: `tests/unit/test_simulation_cache.py`

- [ ] **Step 1: Write the failing tests**

  Create `tests/unit/test_simulation_cache.py`:
  ```python
  import pytest
  from planner.simulation.cache import SimulationCache
  from planner.simulation.client import SimulationResult


  def _result(ttft: float) -> SimulationResult:
      return SimulationResult(
          ttft_p95_ms=ttft, itl_p95_ms=10.0, e2e_p95_ms=1000.0,
          kv_allocation_failure_rate=0.0, preemption_rate=0.0,
          responses_per_sec=9.0,
      )


  @pytest.mark.unit
  def test_cache_miss_returns_none():
      cache = SimulationCache(max_size=10)
      assert cache.get("model", "H100", 1, 512, 256, 9.0, 0) is None


  @pytest.mark.unit
  def test_cache_hit_returns_stored_result():
      cache = SimulationCache(max_size=10)
      result = _result(26.3)
      cache.put("model", "H100", 1, 512, 256, 9.0, 0, result)
      retrieved = cache.get("model", "H100", 1, 512, 256, 9.0, 0)
      assert retrieved is result


  @pytest.mark.unit
  def test_cache_evicts_oldest_entry_when_full():
      cache = SimulationCache(max_size=2)
      r1, r2, r3 = _result(1.0), _result(2.0), _result(3.0)
      cache.put("m", "H100", 1, 512, 256, 1.0, 0, r1)
      cache.put("m", "H100", 1, 512, 256, 2.0, 0, r2)
      cache.put("m", "H100", 1, 512, 256, 3.0, 0, r3)  # evicts r1
      assert cache.get("m", "H100", 1, 512, 256, 1.0, 0) is None
      assert cache.get("m", "H100", 1, 512, 256, 2.0, 0) is r2
      assert cache.get("m", "H100", 1, 512, 256, 3.0, 0) is r3


  @pytest.mark.unit
  def test_cache_key_includes_prefix_tokens():
      cache = SimulationCache(max_size=10)
      r0 = _result(26.0)
      r400 = _result(18.0)
      cache.put("m", "H100", 1, 512, 256, 9.0, 0, r0)
      cache.put("m", "H100", 1, 512, 256, 9.0, 400, r400)
      assert cache.get("m", "H100", 1, 512, 256, 9.0, 0) is r0
      assert cache.get("m", "H100", 1, 512, 256, 9.0, 400) is r400
  ```

- [ ] **Step 2: Run to confirm failure**

  ```bash
  cd src && uv run pytest ../tests/unit/test_simulation_cache.py -v
  ```

  Expected: `ImportError`.

- [ ] **Step 3: Create `src/planner/simulation/cache.py`**

  ```python
  import threading
  from collections import OrderedDict

  from planner.simulation.client import SimulationResult

  _CacheKey = tuple[str, str, int, int, int, float, int]


  class SimulationCache:
      def __init__(self, max_size: int = 2048) -> None:
          self._max = max_size
          self._data: OrderedDict[_CacheKey, SimulationResult] = OrderedDict()
          self._lock = threading.Lock()

      def _key(
          self,
          model: str,
          gpu: str,
          tp: int,
          prompt_tokens: int,
          output_tokens: int,
          qps: float,
          prefix_tokens: int,
      ) -> _CacheKey:
          return (model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)

      def get(
          self,
          model: str,
          gpu: str,
          tp: int,
          prompt_tokens: int,
          output_tokens: int,
          qps: float,
          prefix_tokens: int,
      ) -> SimulationResult | None:
          key = self._key(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)
          with self._lock:
              if key not in self._data:
                  return None
              self._data.move_to_end(key)
              return self._data[key]

      def put(
          self,
          model: str,
          gpu: str,
          tp: int,
          prompt_tokens: int,
          output_tokens: int,
          qps: float,
          prefix_tokens: int,
          result: SimulationResult,
      ) -> None:
          key = self._key(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)
          with self._lock:
              if key in self._data:
                  self._data.move_to_end(key)
              self._data[key] = result
              if len(self._data) > self._max:
                  self._data.popitem(last=False)


  # Module-level singleton: shared across all requests in the same process.
  # Results are deterministic, so cross-request sharing is safe.
  _default_cache = SimulationCache(max_size=2048)


  def get_default_cache() -> SimulationCache:
      return _default_cache
  ```

- [ ] **Step 4: Update `src/planner/simulation/__init__.py`**

  ```python
  from planner.simulation.client import SimulationClient, SimulationResult
  from planner.simulation.cache import SimulationCache, get_default_cache

  __all__ = ["SimulationClient", "SimulationResult", "SimulationCache", "get_default_cache"]
  ```

- [ ] **Step 5: Run tests**

  ```bash
  cd src && uv run pytest ../tests/unit/test_simulation_cache.py -v
  ```

  Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/planner/simulation/cache.py src/planner/simulation/__init__.py \
          tests/unit/test_simulation_cache.py
  git commit -s -m "feat: add simulation LRU cache"
  ```

---

### Task 5: `simulation/router.py`

**Files:**
- Create: `src/planner/simulation/router.py`
- Create: `tests/unit/test_simulation_router.py`

- [ ] **Step 1: Write the failing tests**

  Create `tests/unit/test_simulation_router.py`:
  ```python
  import pytest
  from unittest.mock import MagicMock
  from planner.simulation.router import recommend_cache_affinity, CacheAffinityRecommendation
  from planner.simulation.client import SimulationResult


  def _result(ttft: float) -> SimulationResult:
      return SimulationResult(
          ttft_p95_ms=ttft, itl_p95_ms=10.0, e2e_p95_ms=1000.0,
          kv_allocation_failure_rate=0.0, preemption_rate=0.0,
          responses_per_sec=9.0,
      )


  @pytest.mark.unit
  def test_returns_round_robin_when_no_system_prompt():
      client = MagicMock()
      rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=0)
      assert rec.policy == "round_robin"
      assert rec.simulated_ttft_improvement_pct == 0.0
      client.simulate.assert_not_called()


  @pytest.mark.unit
  def test_returns_cache_aware_when_improvement_exceeds_threshold():
      client = MagicMock()
      # baseline (no prefix): 100ms TTFT; warm (with prefix): 70ms TTFT → 30% improvement
      client.simulate.side_effect = [_result(100.0), _result(70.0)]
      rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=400)
      assert rec.policy == "cache_aware"
      assert rec.simulated_ttft_improvement_pct == pytest.approx(30.0, abs=0.1)


  @pytest.mark.unit
  def test_returns_round_robin_when_improvement_below_threshold():
      client = MagicMock()
      # only 2% improvement — below 5% threshold
      client.simulate.side_effect = [_result(100.0), _result(98.0)]
      rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=400)
      assert rec.policy == "round_robin"


  @pytest.mark.unit
  def test_returns_round_robin_when_simulation_unavailable():
      client = MagicMock()
      client.simulate.return_value = None
      rec = recommend_cache_affinity(client, "m", "H100", 1, 512, 256, 9.0, system_prompt_tokens=400)
      assert rec.policy == "round_robin"
      assert rec.simulated_ttft_improvement_pct == 0.0


  @pytest.mark.unit
  def test_runs_two_simulations_with_correct_prefix_tokens():
      client = MagicMock()
      client.simulate.side_effect = [_result(100.0), _result(70.0)]
      recommend_cache_affinity(client, "my-model", "A100", 2, 1024, 512, 12.0, system_prompt_tokens=600)
      calls = client.simulate.call_args_list
      assert len(calls) == 2
      # first call: prefix_tokens=0 (baseline)
      assert calls[0].kwargs.get("prefix_tokens", calls[0].args[-1] if calls[0].args else None) == 0 or \
             calls[0].args[6] == 0
      # second call: prefix_tokens=600
      assert 600 in calls[1].args or calls[1].kwargs.get("prefix_tokens") == 600
  ```

- [ ] **Step 2: Run to confirm failure**

  ```bash
  cd src && uv run pytest ../tests/unit/test_simulation_router.py -v
  ```

  Expected: `ImportError`.

- [ ] **Step 3: Create `src/planner/simulation/router.py`**

  ```python
  from __future__ import annotations

  import logging
  from concurrent.futures import ThreadPoolExecutor
  from dataclasses import dataclass
  from typing import TYPE_CHECKING, Literal

  if TYPE_CHECKING:
      from planner.simulation.client import SimulationClient

  logger = logging.getLogger(__name__)

  _IMPROVEMENT_THRESHOLD_PCT = 5.0


  @dataclass
  class CacheAffinityRecommendation:
      policy: Literal["cache_aware", "round_robin"]
      reasoning: str
      simulated_ttft_improvement_pct: float


  def recommend_cache_affinity(
      client: SimulationClient,
      model: str,
      gpu: str,
      tp: int,
      prompt_tokens: int,
      output_tokens: int,
      qps: float,
      system_prompt_tokens: int,
  ) -> CacheAffinityRecommendation:
      if system_prompt_tokens <= 0:
          return CacheAffinityRecommendation(
              policy="round_robin",
              reasoning="No shared system prompt — prefix caching provides no benefit.",
              simulated_ttft_improvement_pct=0.0,
          )

      with ThreadPoolExecutor(max_workers=2) as executor:
          f_base = executor.submit(
              client.simulate, model, gpu, tp, prompt_tokens, output_tokens, qps,
              prefix_tokens=0,
          )
          f_warm = executor.submit(
              client.simulate, model, gpu, tp, prompt_tokens, output_tokens, qps,
              prefix_tokens=system_prompt_tokens,
          )
          base = f_base.result()
          warm = f_warm.result()

      if base is None or warm is None or base.ttft_p95_ms <= 0:
          return CacheAffinityRecommendation(
              policy="round_robin",
              reasoning="Cache affinity simulation unavailable — defaulting to round-robin.",
              simulated_ttft_improvement_pct=0.0,
          )

      improvement = (base.ttft_p95_ms - warm.ttft_p95_ms) / base.ttft_p95_ms * 100

      if improvement >= _IMPROVEMENT_THRESHOLD_PCT:
          return CacheAffinityRecommendation(
              policy="cache_aware",
              reasoning=(
                  f"Routing requests to instances with your {system_prompt_tokens}-token "
                  f"system prompt cached reduces TTFT p95 by {improvement:.0f}%. "
                  f"Use prefix-affinity scheduling (e.g., llm-d kv-aware routing)."
              ),
              simulated_ttft_improvement_pct=round(improvement, 1),
          )
      return CacheAffinityRecommendation(
          policy="round_robin",
          reasoning=(
              f"System prompt caching gives <{_IMPROVEMENT_THRESHOLD_PCT:.0f}% TTFT gain "
              f"({improvement:.0f}%) — round-robin is sufficient."
          ),
          simulated_ttft_improvement_pct=round(improvement, 1),
      )
  ```

- [ ] **Step 4: Update `__init__.py`**

  ```python
  from planner.simulation.client import SimulationClient, SimulationResult
  from planner.simulation.cache import SimulationCache, get_default_cache
  from planner.simulation.router import CacheAffinityRecommendation, recommend_cache_affinity

  __all__ = [
      "SimulationClient", "SimulationResult",
      "SimulationCache", "get_default_cache",
      "CacheAffinityRecommendation", "recommend_cache_affinity",
  ]
  ```

- [ ] **Step 5: Run tests**

  ```bash
  cd src && uv run pytest ../tests/unit/test_simulation_router.py -v
  ```

  Expected: all 5 tests PASS.

- [ ] **Step 6: Run all simulation tests**

  ```bash
  cd src && uv run pytest ../tests/unit/test_simulation_client.py \
                          ../tests/unit/test_simulation_cache.py \
                          ../tests/unit/test_simulation_router.py -v
  ```

  Expected: all 17 tests PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add src/planner/simulation/router.py src/planner/simulation/__init__.py \
          tests/unit/test_simulation_router.py
  git commit -s -m "feat: add cache affinity simulation router"
  ```

---

## Chunk 3: Schema changes

### Task 6: Schema additions

**Files:**
- Modify: `src/planner/shared/schemas/specification.py`
- Modify: `src/planner/shared/schemas/recommendation.py`
- Modify: `src/planner/shared/schemas/__init__.py`

- [ ] **Step 1: Read current schema files**

  ```bash
  cat src/planner/shared/schemas/specification.py
  cat src/planner/shared/schemas/recommendation.py
  cat src/planner/shared/schemas/__init__.py
  ```

- [ ] **Step 2: Add `system_prompt_tokens` to `TrafficProfile` in `specification.py`**

  Find the `TrafficProfile` class and add one field:
  ```python
  system_prompt_tokens: int = 0
  ```

  Place it after the existing `expected_qps` field.

- [ ] **Step 3: Add `CacheAffinityRecommendation` to `recommendation.py`**

  Add to the existing imports block (in the local-imports section, after the existing `.intent`, `.specification` imports):
  ```python
  from planner.simulation.router import CacheAffinityRecommendation
  ```

  This creates the dependency chain: `recommendation.py` → `simulation.router` → `simulation.client`. No circular import exists because `simulation/` modules do not import from `shared/schemas/recommendation.py`.

- [ ] **Step 4: Add new fields to `DeploymentRecommendation`**

  Find the `DeploymentRecommendation` class and add these fields (after `predicted_throughput_qps`):
  ```python
  cache_affinity_recommendation: CacheAffinityRecommendation | None = None
  kv_allocation_failure_rate: float = 0.0
  preemption_rate: float = 0.0
  latency_source: Literal["simulation", "database"] = "database"

  @property
  def reliability_status(self) -> Literal["ok", "warning", "critical"]:
      if self.kv_allocation_failure_rate > 0.02:
          return "critical"
      if self.kv_allocation_failure_rate > 0 or self.preemption_rate > 0.05:
          return "warning"
      return "ok"
  ```

  > Pydantic models need `model_config = ConfigDict(...)` to allow computed properties — check if the existing model uses `@property` anywhere; if not, use a `@computed_field` or just a plain `@property` (Pydantic v2 allows both).

- [ ] **Step 5: Export `CacheAffinityRecommendation` from `__init__.py`**

  Add to the exports in `src/planner/shared/schemas/__init__.py`:
  ```python
  from planner.shared.schemas.recommendation import CacheAffinityRecommendation
  ```

- [ ] **Step 6: Run unit tests to catch import errors**

  ```bash
  cd src && uv run pytest ../tests/unit/ -v --tb=short
  ```

  Expected: existing tests still pass, no new import errors.

- [ ] **Step 7: Run type check**

  ```bash
  cd src && uv run mypy planner/shared/schemas/
  ```

  Fix any type errors before committing.

- [ ] **Step 8: Commit**

  ```bash
  git add src/planner/shared/schemas/
  git commit -s -m "feat: add cache affinity and reliability fields to schemas"
  ```

---

## Chunk 4: Backend integration

### Task 7: Populate `system_prompt_tokens` in `TrafficProfileGenerator`

**Files:**
- Modify: `src/planner/specification/traffic_profile.py`
- Modify: `tests/unit/test_traffic_profile.py` (add one test)

- [ ] **Step 1: Read `traffic_profile.py`**

  ```bash
  cat src/planner/specification/traffic_profile.py
  ```

  Find `generate_profile()` — identify where it returns the `TrafficProfile` object.

- [ ] **Step 2: Write a failing test**

  In `tests/unit/test_traffic_profile.py` (or create it if it doesn't exist), add:
  ```python
  @pytest.mark.unit
  def test_generate_profile_sets_system_prompt_tokens_for_chatbot():
      from planner.specification.traffic_profile import TrafficProfileGenerator
      from planner.shared.schemas import DeploymentIntent

      gen = TrafficProfileGenerator(slo_repo=MagicMock())
      intent = DeploymentIntent(use_case="chatbot_conversational", user_count=100)
      profile = gen.generate_profile(intent)
      assert profile.system_prompt_tokens == 400


  @pytest.mark.unit
  def test_generate_profile_defaults_to_zero_for_unknown_use_case():
      from planner.specification.traffic_profile import TrafficProfileGenerator
      from planner.shared.schemas import DeploymentIntent

      gen = TrafficProfileGenerator(slo_repo=MagicMock())
      intent = DeploymentIntent(use_case="unknown_use_case", user_count=100)
      profile = gen.generate_profile(intent)
      assert profile.system_prompt_tokens == 0
  ```

  Run to confirm failure:
  ```bash
  cd src && uv run pytest ../tests/unit/test_traffic_profile.py -k "system_prompt" -v
  ```

- [ ] **Step 3: Add `SYSTEM_PROMPT_TOKEN_DEFAULTS` and populate the field**

  In `src/planner/specification/traffic_profile.py`, add after the imports:
  ```python
  SYSTEM_PROMPT_TOKEN_DEFAULTS: dict[str, int] = {
      "chatbot_conversational": 400,
      "code_completion": 600,
      "code_generation_detailed": 600,
      "rag_qa": 200,
      "long_document_summarization": 0,
      "research_legal_analysis": 0,
  }
  ```

  At the end of `generate_profile()`, before `return profile`:
  ```python
  profile.system_prompt_tokens = SYSTEM_PROMPT_TOKEN_DEFAULTS.get(intent.use_case, 0)
  ```

  Apply the same to `_generate_default_profile()` if it is also a code path that returns a `TrafficProfile`.

- [ ] **Step 4: Run tests**

  ```bash
  cd src && uv run pytest ../tests/unit/test_traffic_profile.py -v
  ```

  Expected: new tests PASS, existing tests unaffected.

- [ ] **Step 5: Commit**

  ```bash
  git add src/planner/specification/traffic_profile.py tests/unit/test_traffic_profile.py
  git commit -s -m "feat: populate system_prompt_tokens in TrafficProfile from use-case defaults"
  ```

---

### Task 8: Parallel simulation in `config_finder.py`

**Files:**
- Modify: `src/planner/recommendation/config_finder.py`
- Create: `tests/unit/test_config_finder_simulation.py`

This is the core integration task. Read `config_finder.py` carefully before starting.

- [ ] **Step 1: Read `config_finder.py` lines 230–550**

  ```bash
  sed -n '230,550p' src/planner/recommendation/config_finder.py
  ```

  Identify:
  - Where `bench.ttft_p95`, `bench.itl_p95`, `bench.e2e_p95` are used to set `predicted_ttft`, `predicted_itl`, `predicted_e2e`
  - Where `DeploymentRecommendation` objects are appended to `all_configs`
  - Where `plan_all_capacities()` ends (the `return` statement)

- [ ] **Step 2: Write failing tests**

  Create `tests/unit/test_config_finder_simulation.py`:
  ```python
  import pytest
  from unittest.mock import MagicMock, patch
  from planner.simulation.client import SimulationResult


  def _sim_result(ttft=26.0, itl=10.0, e2e=3000.0, kv_fail=0.0, preempt=0.0):
      return SimulationResult(
          ttft_p95_ms=ttft, itl_p95_ms=itl, e2e_p95_ms=e2e,
          kv_allocation_failure_rate=kv_fail, preemption_rate=preempt,
          responses_per_sec=9.0, source="simulation",
      )


  @pytest.mark.unit
  def test_simulation_result_overwrites_db_latency(monkeypatch):
      """When simulation succeeds, its latency replaces DB benchmark latency."""
      from planner.recommendation.config_finder import CapacityPlanner

      sim_client = MagicMock()
      sim_client.is_available.return_value = True
      sim_client.simulate.return_value = _sim_result(ttft=20.0, e2e=2000.0)

      planner = CapacityPlanner.__new__(CapacityPlanner)
      # verify the planner uses sim result latency when available
      result = planner._get_latency(sim_client, "model", "H100", 1, 512, 256, 9.0, db_ttft=100, db_e2e=9000)
      assert result.ttft_p95_ms == pytest.approx(20.0)
      assert result.source == "simulation"


  @pytest.mark.unit
  def test_db_latency_used_when_simulation_unavailable(monkeypatch):
      """When simulation returns None, DB latency is preserved."""
      from planner.recommendation.config_finder import CapacityPlanner
      from planner.simulation.client import SimulationResult

      sim_client = MagicMock()
      sim_client.simulate.return_value = None

      planner = CapacityPlanner.__new__(CapacityPlanner)
      result = planner._get_latency(sim_client, "model", "H100", 1, 512, 256, 9.0, db_ttft=100, db_e2e=9000)
      assert result.ttft_p95_ms == pytest.approx(100.0)
      assert result.source == "database"
  ```

  Run to confirm failure:
  ```bash
  cd src && uv run pytest ../tests/unit/test_config_finder_simulation.py -v
  ```

- [ ] **Step 3: Add `SimulationClient` to `CapacityPlanner.__init__`**

  In `config_finder.py`, import at the top:
  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed
  from planner.simulation.client import SimulationClient, SimulationResult
  from planner.simulation.cache import get_default_cache
  from planner.simulation.router import recommend_cache_affinity
  ```

  In `CapacityPlanner.__init__`, add:
  ```python
  self.sim_client = SimulationClient()
  self._sim_cache = get_default_cache()
  ```

- [ ] **Step 4: Add `_get_latency()` helper method to `CapacityPlanner`**

  ```python
  def _get_latency(
      self,
      sim_client: SimulationClient,
      model: str,
      gpu: str,
      tp: int,
      prompt_tokens: int,
      output_tokens: int,
      qps: float,
      db_ttft: float,
      db_e2e: float,
      prefix_tokens: int = 0,
  ) -> SimulationResult:
      cached = self._sim_cache.get(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)
      if cached:
          return cached
      result = sim_client.simulate(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)
      if result is not None:
          self._sim_cache.put(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens, result)
          return result
      # Fallback: wrap DB values in a SimulationResult with source="database"
      return SimulationResult(
          ttft_p95_ms=float(db_ttft or 0),
          itl_p95_ms=0.0,
          e2e_p95_ms=float(db_e2e or 0),
          kv_allocation_failure_rate=0.0,
          preemption_rate=0.0,
          responses_per_sec=0.0,
          source="database",
      )
  ```

- [ ] **Step 5: Integrate simulation into the benchmark loop in `plan_all_capacities()`**

  Find the block where `predicted_ttft = int(bench.ttft_p95)` is set (around line 365 based on earlier reading). Replace with:

  ```python
  # Get latency from simulation (falls back to DB values)
  qps = traffic_profile.expected_qps or 1.0
  sim_result = self._get_latency(
      self.sim_client,
      bench.model_hf_repo,
      bench.hardware,
      bench.hardware_count,  # bench.hardware_count = tensor_parallelism = per-replica GPU count (they're equal in bench schema)
      traffic_profile.prompt_tokens,
      traffic_profile.output_tokens,
      qps,
      db_ttft=bench.ttft_p95 or 0,
      db_e2e=bench.e2e_p95 or 0,
  )
  predicted_ttft = int(sim_result.ttft_p95_ms)
  predicted_itl = int(sim_result.itl_p95_ms) if sim_result.itl_p95_ms else int(bench.itl_p95 or 0)
  predicted_e2e = int(sim_result.e2e_p95_ms)
  ```

  After building the `DeploymentRecommendation` object, add the new fields:
  ```python
  rec.kv_allocation_failure_rate = sim_result.kv_allocation_failure_rate
  rec.preemption_rate = sim_result.preemption_rate
  rec.latency_source = sim_result.source
  ```

  > Check the exact variable name used for the recommendation object (it may be `config`, `rec`, or `recommendation`).

- [ ] **Step 6: Add cache affinity enrichment at the end of `plan_all_capacities()`**

  Just before the `return` statement, add:
  ```python
  # Enrich top-3 balanced candidates with cache affinity simulation
  system_prompt_tokens = getattr(traffic_profile, "system_prompt_tokens", 0)
  if system_prompt_tokens > 0 and self.sim_client.is_available():
      top3 = sorted(all_configs, key=lambda r: r.scores.balanced_score if r.scores else 0, reverse=True)[:3]
      qps = traffic_profile.expected_qps or 1.0
      with ThreadPoolExecutor(max_workers=min(3, len(top3))) as executor:
          futures = {
              executor.submit(
                  recommend_cache_affinity,
                  self.sim_client,
                  r.model_id,
                  r.gpu_config.gpu_type,
                  r.gpu_config.tensor_parallel,
                  traffic_profile.prompt_tokens,
                  traffic_profile.output_tokens,
                  qps,
                  system_prompt_tokens,
              ): r
              for r in top3
          }
          for fut in as_completed(futures):
              rec = futures[fut]
              try:
                  rec.cache_affinity_recommendation = fut.result()
              except Exception as e:
                  logger.warning("Cache affinity simulation failed: %s", e)
  ```

  > Field names confirmed from schema: `r.model_id` (str), `r.gpu_config.gpu_type` (str), `r.gpu_config.tensor_parallel` (int) — all valid on `DeploymentRecommendation` and `GPUConfig`.

- [ ] **Step 7: Run tests**

  ```bash
  cd src && uv run pytest ../tests/unit/test_config_finder_simulation.py -v
  cd src && uv run pytest ../tests/unit/ -v --tb=short
  ```

  Expected: all tests PASS.

- [ ] **Step 8: Smoke test with the running backend**

  ```bash
  make start
  curl -s -X POST http://localhost:8000/api/v1/recommend \
    -H "Content-Type: application/json" \
    -d '{"message": "I need a chatbot for 500 users"}' | python3 -m json.tool | head -60
  ```

  Look for `latency_source`, `kv_allocation_failure_rate`, `preemption_rate` fields in the response.

- [ ] **Step 9: Commit**

  ```bash
  git add src/planner/recommendation/config_finder.py \
          tests/unit/test_config_finder_simulation.py
  git commit -s -m "feat: integrate inference-sim parallel simulation into capacity planner"
  ```

---

## Chunk 5: API endpoint + UI

### Task 9: `POST /api/v1/explore-config`

**Files:**
- Create: `src/planner/api/routes/explore.py`
- Modify: `src/planner/api/routes/__init__.py`
- Modify: `src/planner/api/app.py`
- Create: `tests/unit/test_explore_endpoint.py`

- [ ] **Step 1: Write the failing test**

  Create `tests/unit/test_explore_endpoint.py`:
  ```python
  import pytest
  from fastapi.testclient import TestClient
  from unittest.mock import patch, MagicMock
  from planner.api.app import create_app
  from planner.simulation.client import SimulationResult


  @pytest.fixture
  def client():
      app = create_app()
      return TestClient(app)


  def _sim_result():
      return SimulationResult(
          ttft_p95_ms=26.3, itl_p95_ms=10.7, e2e_p95_ms=3856.1,
          kv_allocation_failure_rate=0.0, preemption_rate=0.0,
          responses_per_sec=9.91, source="simulation",
      )


  @pytest.mark.unit
  def test_explore_config_returns_simulation_metrics(client):
      with patch("planner.api.routes.explore.SimulationClient") as MockClient:
          mock_instance = MagicMock()
          mock_instance.simulate.return_value = _sim_result()
          MockClient.return_value = mock_instance

          response = client.post("/api/v1/explore-config", json={
              "model": "meta-llama/llama-3.1-8b-instruct",
              "gpu_type": "H100",
              "tensor_parallelism": 1,
              "gpu_count": 1,
              "prompt_tokens": 512,
              "output_tokens": 256,
              "qps": 9.0,
              "replicas": 1,
          })

      assert response.status_code == 200
      data = response.json()
      assert data["simulation"]["ttft_p95_ms"] == pytest.approx(26.3)
      assert "monthly_cost_usd" in data
      assert "reliability_status" in data


  @pytest.mark.unit
  def test_explore_config_returns_503_when_binary_unavailable(client):
      with patch("planner.api.routes.explore.SimulationClient") as MockClient:
          mock_instance = MagicMock()
          mock_instance.simulate.return_value = None
          mock_instance.is_available.return_value = False
          MockClient.return_value = mock_instance

          response = client.post("/api/v1/explore-config", json={
              "model": "meta-llama/llama-3.1-8b-instruct",
              "gpu_type": "H100",
              "tensor_parallelism": 1,
              "gpu_count": 1,
              "prompt_tokens": 512,
              "output_tokens": 256,
              "qps": 9.0,
              "replicas": 1,
          })

      assert response.status_code == 503


  @pytest.mark.unit
  def test_explore_config_returns_400_for_zero_qps(client):
      response = client.post("/api/v1/explore-config", json={
          "model": "m", "gpu_type": "H100", "tensor_parallelism": 1,
          "gpu_count": 1, "prompt_tokens": 512, "output_tokens": 256,
          "qps": 0.0, "replicas": 1,
      })
      assert response.status_code == 400
  ```

  Run to confirm failure:
  ```bash
  cd src && uv run pytest ../tests/unit/test_explore_endpoint.py -v
  ```

- [ ] **Step 2: Create `src/planner/api/routes/explore.py`**

  First read `src/planner/api/routes/recommendation.py` to understand the route file pattern:
  ```bash
  head -40 src/planner/api/routes/recommendation.py
  ```

  Then create:
  ```python
  import logging
  from typing import Literal

  from fastapi import APIRouter, HTTPException
  from pydantic import BaseModel, Field

  from planner.simulation.client import SimulationClient, SimulationResult
  from planner.simulation.cache import get_default_cache
  from planner.simulation.router import CacheAffinityRecommendation, recommend_cache_affinity

  logger = logging.getLogger(__name__)
  router = APIRouter(prefix="/api/v1")

  # GPU pricing ($/hour) — same source as config_finder.py; keep in sync
  _GPU_HOURLY_RATES: dict[str, float] = {
      "H100": 8.0, "A100-80GB": 4.5, "A100-40GB": 3.0,
      "L4": 0.7, "L40S": 2.0, "T4": 0.35,
  }
  _HOURS_PER_MONTH = 730


  class ExploreConfigRequest(BaseModel):
      model: str
      gpu_type: str
      tensor_parallelism: int = Field(ge=1)
      gpu_count: int = Field(ge=1)
      prompt_tokens: int = Field(ge=1)
      output_tokens: int = Field(ge=1)
      qps: float = Field(gt=0, description="Must be > 0")
      replicas: int = Field(ge=1)
      prefix_tokens: int = Field(default=0, ge=0)


  class ExploreConfigResponse(BaseModel):
      simulation: dict
      monthly_cost_usd: float
      reliability_status: Literal["ok", "warning", "critical"]
      cache_affinity_recommendation: CacheAffinityRecommendation | None


  def _reliability_status(
      kv_rate: float, preempt_rate: float
  ) -> Literal["ok", "warning", "critical"]:
      if kv_rate > 0.02:
          return "critical"
      if kv_rate > 0 or preempt_rate > 0.05:
          return "warning"
      return "ok"


  @router.post("/explore-config", response_model=ExploreConfigResponse)
  def explore_config(req: ExploreConfigRequest) -> ExploreConfigResponse:
      client = SimulationClient()
      cache = get_default_cache()

      cached = cache.get(
          req.model, req.gpu_type, req.tensor_parallelism,
          req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens,
      )
      sim = cached
      if sim is None:
          sim = client.simulate(
              req.model, req.gpu_type, req.tensor_parallelism,
              req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens,
          )
          if sim is not None:
              cache.put(
                  req.model, req.gpu_type, req.tensor_parallelism,
                  req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens, sim,
              )

      if sim is None:
          raise HTTPException(
              status_code=503,
              detail="Simulation unavailable. Recommendation card values remain valid.",
          )

      hourly = _GPU_HOURLY_RATES.get(req.gpu_type, 2.0)
      monthly_cost = req.gpu_count * req.tensor_parallelism * req.replicas * hourly * _HOURS_PER_MONTH

      affinity: CacheAffinityRecommendation | None = None
      if req.prefix_tokens > 0:
          affinity = recommend_cache_affinity(
              client, req.model, req.gpu_type, req.tensor_parallelism,
              req.prompt_tokens, req.output_tokens, req.qps, req.prefix_tokens,
          )

      return ExploreConfigResponse(
          simulation={
              "ttft_p95_ms": sim.ttft_p95_ms,
              "itl_p95_ms": sim.itl_p95_ms,
              "e2e_p95_ms": sim.e2e_p95_ms,
              "kv_allocation_failure_rate": sim.kv_allocation_failure_rate,
              "preemption_rate": sim.preemption_rate,
              "responses_per_sec": sim.responses_per_sec,
          },
          monthly_cost_usd=round(monthly_cost, 2),
          reliability_status=_reliability_status(
              sim.kv_allocation_failure_rate, sim.preemption_rate
          ),
          cache_affinity_recommendation=affinity,
      )
  ```

- [ ] **Step 3: Register the router**

  In `src/planner/api/routes/__init__.py`, add:
  ```python
  from planner.api.routes.explore import router as explore_router
  ```

  In `src/planner/api/app.py`, include the router alongside existing ones:
  ```python
  from planner.api.routes import explore_router
  # ...
  app.include_router(explore_router)
  ```

- [ ] **Step 4: Run tests**

  ```bash
  cd src && uv run pytest ../tests/unit/test_explore_endpoint.py -v
  ```

  Expected: all 3 tests PASS.

- [ ] **Step 5: Verify in Swagger UI**

  ```bash
  make start
  open http://localhost:8000/docs
  ```

  Find `POST /api/v1/explore-config`. Test with a valid payload.

- [ ] **Step 6: Commit**

  ```bash
  git add src/planner/api/routes/explore.py \
          src/planner/api/routes/__init__.py \
          src/planner/api/app.py \
          tests/unit/test_explore_endpoint.py
  git commit -s -m "feat: add /api/v1/explore-config endpoint for scenario simulation"
  ```

---

### Task 10: UI — reliability + cache affinity rows + source badge on cards

**Files:**
- Modify: `ui/components/recommendations.py`

- [ ] **Step 1: Read the current card rendering code**

  ```bash
  cat ui/components/recommendations.py
  ```

  Find `_render_category_card()` and the metrics line that shows `TTFT: Xms | ITL: Yms | E2E: Zms`.

- [ ] **Step 2: Add source badge and reliability row to the card**

  In `_render_category_card()`, after the existing metrics display, add:

  ```python
  # Source badge
  latency_source = rec.get("latency_source", "database")
  source_badge = "🔬 sim" if latency_source == "simulation" else "📊 db"

  # Reliability row
  kv_fail_rate = rec.get("kv_allocation_failure_rate", 0.0)
  preempt_rate = rec.get("preemption_rate", 0.0)
  reliability_status = rec.get("reliability_status", "ok")  # pre-computed or compute inline

  if kv_fail_rate > 0.02:
      reliability_icon = "🔴"
      reliability_text = f"{kv_fail_rate*100:.1f}% KV allocation failures — insufficient GPU memory"
  elif kv_fail_rate > 0 or preempt_rate > 0.05:
      reliability_icon = "⚠️"
      reliability_text = f"{preempt_rate*100:.1f}% preemptions — consider adding 1 replica"
  else:
      reliability_icon = "✅"
      reliability_text = f"No KV failures · {preempt_rate*100:.1f}% preemptions"

  # Cache affinity row
  cache_rec = rec.get("cache_affinity_recommendation")
  cache_line = ""
  if cache_rec:
      policy = cache_rec.get("policy", "round_robin")
      improvement = cache_rec.get("simulated_ttft_improvement_pct", 0.0)
      if policy == "cache_aware" and improvement > 0:
          cache_line = f"Cache Affinity   cache-aware → {improvement:.0f}% faster TTFT p95  (simulated)"
      else:
          cache_line = "Cache Affinity   round-robin sufficient"
  ```

  Then render these in the card layout. Exactly how depends on the current layout — follow the existing pattern (column-based or markdown). A simple approach:

  ```python
  st.caption(f"{source_badge}  |  {reliability_icon} {reliability_text}")
  if cache_line:
      st.caption(cache_line)
  ```

- [ ] **Step 3: Verify visually**

  ```bash
  make start
  open http://localhost:8501
  ```

  Submit a recommendation request. Verify each card shows:
  - `🔬 sim` or `📊 db` badge
  - Reliability row with icon
  - Cache affinity row (if applicable)

- [ ] **Step 4: Commit**

  ```bash
  git add ui/components/recommendations.py
  git commit -s -m "feat: add reliability and cache affinity rows to recommendation cards"
  ```

---

### Task 11: UI — Explore panel

**Files:**
- Modify: `ui/components/recommendations.py`

- [ ] **Step 1: Add the Explore expander to each card**

  In the card rendering function, after the reliability/cache rows, add an expander:

  ```python
  with st.expander("🔬 Explore scenarios"):
      _render_explore_panel(rec, backend_url)
  ```

- [ ] **Step 2: Implement `_render_explore_panel()`**

  Add this function to `ui/components/recommendations.py`:

  `DeploymentRecommendation` serializes to a dict with nested `gpu_config` and `traffic_profile` keys. The correct access patterns are confirmed from schema inspection:
  - `rec["model_id"]` — flat
  - `rec["gpu_config"]["gpu_type"]`, `["gpu_count"]`, `["tensor_parallel"]`, `["replicas"]` — nested under `gpu_config`
  - `rec["traffic_profile"]["expected_qps"]`, `["prompt_tokens"]`, `["output_tokens"]` — nested under `traffic_profile`

  ```python
  import requests  # already used in the UI

  def _reliability_emoji(status: str) -> str:
      return {"ok": "✅", "warning": "⚠️", "critical": "🔴"}.get(status, "❓")


  def _render_explore_panel(rec: dict, backend_url: str) -> None:
      gpu_cfg = rec.get("gpu_config") or {}
      traffic = rec.get("traffic_profile") or {}

      rec_id = f"{rec.get('model_id', '')}_{gpu_cfg.get('gpu_type', '')}_{gpu_cfg.get('tensor_parallel', 1)}"
      state_key = f"explore_scenarios_{rec_id}"

      if state_key not in st.session_state:
          st.session_state[state_key] = []

      base_qps = float(traffic.get("expected_qps") or 9.0)
      base_replicas = int(gpu_cfg.get("replicas") or 1)

      # Preset buttons
      col1, col2, col3, col4 = st.columns(4)
      run_preset = None
      with col1:
          if st.button("Recommended", key=f"preset_base_{rec_id}"):
              run_preset = {"label": "Recommended", "qps": base_qps, "replicas": base_replicas}
      with col2:
          if st.button("2× Traffic", key=f"preset_2x_{rec_id}"):
              run_preset = {"label": "2× Traffic", "qps": base_qps * 2, "replicas": base_replicas + 1}
      with col3:
          if st.button("Peak Load", key=f"preset_peak_{rec_id}"):
              run_preset = {"label": "Peak Load", "qps": base_qps * 3, "replicas": base_replicas + 2}
      with col4:
          show_custom = st.toggle("Custom", key=f"custom_toggle_{rec_id}")

      # Custom sliders
      if show_custom:
          custom_qps = st.slider(
              "QPS", min_value=base_qps * 0.25, max_value=base_qps * 2.0,
              value=base_qps, step=0.5, key=f"custom_qps_{rec_id}"
          )
          custom_replicas = st.number_input(
              "Replicas", min_value=1, max_value=base_replicas + 4,
              value=base_replicas, key=f"custom_replicas_{rec_id}"
          )
          if st.button("Simulate", key=f"custom_sim_{rec_id}"):
              run_preset = {"label": "Custom", "qps": custom_qps, "replicas": custom_replicas}

      # Run a preset scenario
      if run_preset is not None:
          with st.spinner(f"Simulating {run_preset['label']}..."):
              payload = {
                  "model": rec.get("model_id"),
                  "gpu_type": gpu_cfg.get("gpu_type"),
                  "tensor_parallelism": gpu_cfg.get("tensor_parallel", 1),
                  "gpu_count": gpu_cfg.get("gpu_count", 1),
                  "prompt_tokens": traffic.get("prompt_tokens", 512),
                  "output_tokens": traffic.get("output_tokens", 256),
                  "qps": run_preset["qps"],
                  "replicas": run_preset["replicas"],
              }
              try:
                  resp = requests.post(f"{backend_url}/api/v1/explore-config", json=payload, timeout=15)
                  if resp.status_code == 200:
                      data = resp.json()
                      row = {
                          "Scenario": run_preset["label"],
                          "QPS": f"{run_preset['qps']:.1f}",
                          "Replicas": run_preset["replicas"],
                          "TTFT p95": f"{data['simulation']['ttft_p95_ms']:.0f}ms",
                          "Cost/mo": f"${data['monthly_cost_usd']:,.0f}",
                          "Reliability": _reliability_emoji(data["reliability_status"]),
                      }
                      scenarios = st.session_state[state_key]
                      if len(scenarios) >= 10:
                          non_base = [s for s in scenarios if s["Scenario"] != "Recommended"]
                          if non_base:
                              scenarios.remove(non_base[0])
                      scenarios.append(row)
                      st.session_state[state_key] = scenarios

                      if data["reliability_status"] in ("warning", "critical"):
                          st.info(
                              f"⚠️ {run_preset['label']} shows {data['reliability_status']} reliability. "
                              f"Try adding 1 more replica."
                          )
                  elif resp.status_code == 503:
                      st.warning("Simulation binary unavailable. Install Go and run `make setup-inference-sim`.")
                  else:
                      st.error(f"Simulation failed: {resp.status_code}")
              except requests.exceptions.RequestException as e:
                  st.error(f"Could not reach backend: {e}")

      # Results table
      scenarios = st.session_state.get(state_key, [])
      if scenarios:
          import pandas as pd
          st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)
  ```

- [ ] **Step 3: Pass `backend_url` to the card rendering function**

  First update the `_render_category_card()` function signature (at the top of the function) to accept `backend_url`:
  ```python
  def _render_category_card(rec: dict, backend_url: str = "http://localhost:8000") -> None:
  ```

  Then find every call to `_render_category_card(rec)` in the UI and update it:
  ```python
  backend_url = st.session_state.get("backend_url", "http://localhost:8000")
  _render_category_card(rec, backend_url=backend_url)
  ```

- [ ] **Step 4: Verify the Explore panel end-to-end**

  ```bash
  make start
  open http://localhost:8501
  ```

  1. Submit: "I need a chatbot for 500 users"
  2. Expand a recommendation card → click "2× Traffic" → verify a row appears in the table
  3. Click "Peak Load" → verify second row appears
  4. Click "Custom →" → move QPS slider → click "Simulate" → verify custom row appears

- [ ] **Step 5: Run all unit tests**

  ```bash
  cd src && uv run pytest ../tests/unit/ -v --tb=short
  ```

  Expected: all tests PASS.

- [ ] **Step 6: Run linter and type check**

  ```bash
  make lint
  make typecheck
  ```

  Fix any errors before committing.

- [ ] **Step 7: Final commit**

  ```bash
  git add ui/components/recommendations.py
  git commit -s -m "feat: add Explore scenarios panel to recommendation cards"
  ```

---

## Done

After all tasks are complete:

```bash
cd src && uv run pytest ../tests/unit/ -v
make lint
make typecheck
```

All tests should pass. The planner now:
1. Calls inference-sim live for latency predictions on any traffic profile
2. Falls back to the static DB benchmark when the binary is unavailable
3. Shows a `🔬 sim` / `📊 db` badge on each card
4. Shows reliability status (KV failures, preemptions) per card
5. Shows cache affinity recommendation with a simulated TTFT improvement % for top-3 balanced candidates
6. Lets users run 2× Traffic / Peak Load / Custom scenario simulations from an Explore panel per card
