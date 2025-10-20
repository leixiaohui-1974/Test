from __future__ import annotations

import json

import pytest

from hydraulic_model.solvers import FixedSpeedPumpBoundary, PassiveGateBoundary
from hydraulic_model.workflows import InternalBoundaryConfigError, load_internal_boundaries


def test_load_internal_boundaries_from_list(tmp_path) -> None:
    config = [
        {
            "id": "gate-mid",
            "type": "passive_gate",
            "face_index": 3,
            "width": 5.0,
            "invert_elevation": 12.5,
            "discharge_coeff": 0.58,
            "default_opening": 0.4,
            "state": {"opening": 0.45, "discharge": 3.2, "metadata": {"tag": "init"}},
        }
    ]
    path = tmp_path / "boundaries.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    bundle = load_internal_boundaries(path)
    assert len(bundle.boundaries) == 1
    assert isinstance(bundle.boundaries[0], PassiveGateBoundary)
    gate = bundle.boundaries[0]
    assert gate.boundary_id == "gate-mid"
    assert gate.face_index == 3
    assert gate.default_opening == pytest.approx(0.4)

    state = bundle.states["gate-mid"]
    assert state.opening == pytest.approx(0.45)
    assert state.discharge == pytest.approx(3.2)
    assert state.metadata["tag"] == "init"


def test_load_internal_boundaries_requires_supported_type(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"id": "x", "type": "unknown"}]), encoding="utf-8")
    with pytest.raises(InternalBoundaryConfigError):
        load_internal_boundaries(path)


def test_load_internal_boundaries_deduplicates_ids(tmp_path) -> None:
    config = [
        {
            "id": "gate-1",
            "face_index": 0,
            "width": 2.0,
            "invert_elevation": 0.0,
        },
        {
            "id": "gate-1",
            "face_index": 1,
            "width": 2.0,
            "invert_elevation": 0.0,
        },
    ]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(InternalBoundaryConfigError):
        load_internal_boundaries(path)


def test_load_internal_boundaries_parses_pump(tmp_path) -> None:
    config = {
        "boundaries": [
            {
                "id": "pump-a",
                "type": "fixed_speed_pump",
                "face_index": 2,
                "rated_discharge": 4.5,
                "rated_head": 1.8,
                "shutoff_head": 2.2,
                "efficiency_curve": [[0.0, 0.5], [4.5, 0.82], [6.0, 0.78]],
                "state": {"pump_state": "on", "metadata": {"unit": "a1"}},
                "control": {"mode": "schedule", "entries": [[0, "on"], [1800, "off"]]},
            }
        ]
    }
    path = tmp_path / "pump.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    bundle = load_internal_boundaries(path)
    assert len(bundle.boundaries) == 1
    boundary = bundle.boundaries[0]
    assert isinstance(boundary, FixedSpeedPumpBoundary)
    assert boundary.face_index == 2
    assert boundary.efficiency_curve == ((0.0, 0.5), (4.5, 0.82), (6.0, 0.78))
    assert boundary.control_mode == "schedule"
    assert boundary.control_schedule == ((0.0, "on"), (1800.0, "off"))
    state = bundle.states["pump-a"]
    assert state.pump_state == "on"
    assert state.metadata["unit"] == "a1"


def test_load_internal_boundaries_other_control_modes(tmp_path) -> None:
    config = [
        {
            "id": "pump-level",
            "type": "fixed_speed_pump",
            "face_index": 1,
            "rated_discharge": 3.0,
            "rated_head": 0.8,
            "control": {"mode": "level", "on_level": 2.4, "off_level": 2.1, "sense": "downstream"},
        },
        {
            "id": "pump-cycle",
            "type": "fixed_speed_pump",
            "face_index": 2,
            "rated_discharge": 2.5,
            "rated_head": 0.9,
            "control": {"mode": "cycle", "on_duration": 600, "off_duration": 300, "phase": 60},
        },
    ]

    path = tmp_path / "control.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    bundle = load_internal_boundaries(path)
    pump_level = bundle.boundaries[0]
    pump_cycle = bundle.boundaries[1]
    assert pump_level.control_mode == "level"
    assert pump_level.control_levels == (2.4, 2.1, "downstream")
    assert pump_cycle.control_mode == "cycle"
    assert pump_cycle.control_cycle == (600.0, 300.0, 60.0)
