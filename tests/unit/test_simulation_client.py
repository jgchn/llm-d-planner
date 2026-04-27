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
def test_simulate_constructs_correct_command():
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = SAMPLE_JSON

    with patch("subprocess.run", return_value=mock_proc) as mock_run, \
         patch("os.path.isfile", return_value=True), \
         patch("os.access", return_value=True):
        client = SimulationClient(bin_path="/fake/binary")
        client.simulate("meta-llama/llama-3.1-8b-instruct", "H100", 1, 512, 256, 9.0)

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/fake/binary"
    assert "run" in cmd
    assert "--model" in cmd
    assert "meta-llama/llama-3.1-8b-instruct" in cmd
    assert "--hardware" in cmd
    assert "H100" in cmd
    assert "--tp" in cmd
    assert "1" in cmd
    assert "--prompt-tokens" in cmd
    assert "512" in cmd
    assert "--output-tokens" in cmd
    assert "256" in cmd
    assert "--rate" in cmd
    assert "9.0" in cmd
    assert "--max-prompts" in cmd
    assert "--output" in cmd
    assert "json" in cmd
