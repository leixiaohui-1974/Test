from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import matplotlib.pyplot as plt

from hydraulic_model.steady import SteadyResults
from hydraulic_model.unsteady import UnsteadyResults
from hydraulic_model.solvers import InternalBoundaryState
from hydraulic_model.visualization import (
    animate_unsteady_propagation,
    plot_steady_profile,
    plot_unsteady_heatmap,
    plot_unsteady_timeseries,
)


def _scenario_prefix(scenario: str) -> str:
    return scenario.replace(" ", "_")


def export_steady_results(results: SteadyResults, output_dir: Path, scenario: str) -> None:
    """Persist steady results和概要图，并在文件名中标注场景。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _scenario_prefix(scenario)
    df = results.to_dataframe()
    df.to_csv(output_dir / f"steady_profile_{prefix}.csv", index=False)
    print("稳态水力曲线 (前几行):")
    print(df.head())
    plot_steady_profile(results, output_dir / f"steady_profile_{prefix}.png")


def export_unsteady_overview(
    results: UnsteadyResults,
    bed_elevations: np.ndarray,
    output_dir: Path,
    *,
    scenario: str,
    solver_name: str,
) -> None:
    """导出主基线非恒定结果，文件名包含场景与求解器名称。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _scenario_prefix(scenario)
    base_name = f"unsteady_{prefix}_{solver_name}"
    tidy = results.to_long_dataframe()
    tidy.to_csv(output_dir / f"{base_name}_timeseries.csv", index=False)
    print("非恒定流时序数据 (前几行):")
    print(tidy.head())

    num_stations = results.depths.shape[1]
    mid_index = min(num_stations - 1, max(0, num_stations // 2))
    selected_indices = sorted({0, mid_index, num_stations - 1})
    plot_unsteady_timeseries(results, selected_indices, output_dir / f"{base_name}_hydrographs")
    plot_unsteady_heatmap(results, output_dir / f"{base_name}_depth_heatmap.png", field="depth")
    plot_unsteady_heatmap(
        results, output_dir / f"{base_name}_discharge_heatmap.png", field="discharge"
    )
    animate_unsteady_propagation(
        results,
        bed_elevations,
        output_dir / f"{base_name}_propagation.gif",
        max_frames=200,
    )


def export_unsteady_diagnostics(
    results: UnsteadyResults,
    bed_elevations: np.ndarray,
    output_dir: Path,
    *,
    scenario: str,
    solver_name: str,
    boundary_states: Mapping[str, InternalBoundaryState] | None = None,
) -> None:
    """导出辅助求解器的非恒定结果，文件名包含场景与求解器名称。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _scenario_prefix(scenario)
    base_name = f"unsteady_{prefix}_{solver_name}"
    tidy = results.to_long_dataframe()
    tidy.to_csv(output_dir / f"{base_name}_timeseries.csv", index=False)
    animate_unsteady_propagation(
        results,
        bed_elevations,
        output_dir / f"{base_name}_propagation.gif",
        max_frames=200,
    )
    if boundary_states:
        export_boundary_metrics(boundary_states, output_dir, base_name)


def export_boundary_metrics(
    boundary_states: Mapping[str, InternalBoundaryState],
    output_dir: Path,
    base_name: str,
) -> None:
    summary_rows = []
    for boundary_id, state in boundary_states.items():
        metadata = state.metadata
        if not isinstance(metadata, dict):
            continue
        history = metadata.get("history")
        if not isinstance(history, list) or not history:
            continue
        csv_path = output_dir / f"{base_name}_{boundary_id}_boundary.csv"
        fieldnames = [
            "time",
            "discharge",
            "upstream_stage",
            "downstream_stage",
            "pump_on",
            "efficiency",
            "input_power_kw",
            "hydraulic_power_kw",
            "energy_kwh",
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for sample in history:
                writer.writerow({key: sample.get(key, "") for key in fieldnames})

        summary_rows.append(
            {
                "boundary_id": boundary_id,
                "final_energy_kwh": metadata.get("energy_kwh", 0.0),
                "final_discharge": state.discharge,
                "pump_state": state.pump_state,
            }
        )

        plot_boundary_history(history, output_dir / f"{base_name}_{boundary_id}_metrics.png")

    if summary_rows:
        summary_path = output_dir / f"{base_name}_boundary_summary.csv"
        fieldnames = ["boundary_id", "final_energy_kwh", "final_discharge", "pump_state"]
        with summary_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)


def plot_boundary_history(history: list[dict[str, float]], output_path: Path) -> None:
    times = np.array([entry.get("time", 0.0) for entry in history], dtype=float)
    discharge = np.array([entry.get("discharge", 0.0) for entry in history], dtype=float)
    upstream_stage = np.array([entry.get("upstream_stage", 0.0) for entry in history], dtype=float)
    downstream_stage = np.array([entry.get("downstream_stage", 0.0) for entry in history], dtype=float)
    efficiency = np.array([entry.get("efficiency", 0.0) for entry in history], dtype=float)
    power_kw = np.array([entry.get("input_power_kw", 0.0) for entry in history], dtype=float)
    energy_kwh = np.array([entry.get("energy_kwh", 0.0) for entry in history], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(times, discharge, label="Discharge [m³/s]")
    axes[0].set_ylabel("Discharge")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, upstream_stage, label="Upstream Stage")
    axes[1].plot(times, downstream_stage, label="Downstream Stage")
    axes[1].set_ylabel("Stage [m]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, power_kw, label="Input Power [kW]")
    axes[2].plot(times, efficiency, label="Efficiency", linestyle="--")
    axes[2].set_ylabel("Power / Efficiency")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, energy_kwh, label="Energy [kWh]")
    axes[3].set_ylabel("Energy [kWh]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
