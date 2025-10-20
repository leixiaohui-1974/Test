from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_hll_cli_accepts_internal_boundaries(tmp_path):
    config = Path("hydraulic_model/examples/gate_pump_chain.json").resolve()
    output_dir = tmp_path / "outputs"
    cmd = [
        sys.executable,
        "-m",
        "hydraulic_model.cli.hll_muscl",
        "--scenario",
        "open",
        "--total-time",
        "600",
        "--dt",
        "1.0",
        "--internal-boundaries",
        str(config),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)

    scenario_dir = output_dir / "open"
    timeseries = scenario_dir / "unsteady_open_hll_timeseries.csv"
    assert timeseries.exists()
    summary = scenario_dir / "unsteady_open_hll_boundary_summary.csv"
    assert summary.exists()
