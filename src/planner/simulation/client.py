import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# client.py is at src/planner/simulation/client.py
# Going up 4 dirs from that file's directory lands at the workspace root,
# where inference-sim/ is a sibling of llm-d-planner/. Override with
# INFERENCE_SIM_BIN env var if your layout differs.
_SIM_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../inference-sim")
)
_DEFAULT_BIN = os.path.join(_SIM_DIR, "simulation_worker")


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

        sim_dir = os.environ.get(
            "INFERENCE_SIM_DIR",
            os.path.dirname(self.bin_path),
        )
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
            "--coeffs-filepath", os.path.join(sim_dir, "coefficients.yaml"),
            "--workloads-filepath", os.path.join(sim_dir, "workloads.yaml"),
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
