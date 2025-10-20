from __future__ import annotations

import math

import pytest

from hydraulic_model.geometry import ChannelProfile, CrossSection
from hydraulic_model.hydraulics import GRAVITY
from hydraulic_model.network import HydraulicEdge, HydraulicNetwork, HydraulicNode
from hydraulic_model.solvers import (
    BoundaryEvaluation,
    FixedSpeedPumpBoundary,
    HLLMusclSolver,
    InternalBoundaryState,
    NodeState,
    PassiveGateBoundary,
    PreissmannImplicitSolver,
)


class DummyBoundary:
    boundary_id = "gate-1"
    upstream_node = "up"
    downstream_node = "down"

    def evaluate(
        self,
        time: float,
        upstream: NodeState,
        downstream: NodeState,
        state: InternalBoundaryState,
    ) -> BoundaryEvaluation:
        discharge = state.discharge + (upstream.stage - downstream.stage)
        return BoundaryEvaluation(
            discharge=discharge,
            upstream_stage=upstream.stage,
            downstream_stage=downstream.stage,
            metadata={"time": time},
        )

    def update_state(
        self,
        evaluation: BoundaryEvaluation,
        state: InternalBoundaryState,
    ) -> InternalBoundaryState:
        updated = InternalBoundaryState(
            discharge=evaluation.discharge,
            opening=state.opening,
            pump_state=state.pump_state,
            metadata=dict(state.metadata),
        )
        updated.metadata["last_discharge"] = evaluation.discharge
        return updated


def test_network_validation_detects_missing_nodes() -> None:
    network = HydraulicNetwork(
        nodes={"up": HydraulicNode(node_id="up", kind="boundary")},
        edges={"reach": HydraulicEdge(edge_id="reach", upstream_node="up", downstream_node="down")},
    )
    with pytest.raises(ValueError):
        network.validate()


def test_internal_boundary_state_uses_independent_metadata_maps() -> None:
    state_a = InternalBoundaryState()
    state_b = InternalBoundaryState()
    state_a.metadata["tag"] = "a"
    assert "tag" not in state_b.metadata


def test_dummy_boundary_round_trip() -> None:
    boundary = DummyBoundary()
    upstream = NodeState(stage=1.8, discharge=2.0)
    downstream = NodeState(stage=0.9, discharge=2.0)
    state = InternalBoundaryState(discharge=1.2, opening=0.5, metadata={"tag": "baseline"})

    evaluation = boundary.evaluate(time=12.0, upstream=upstream, downstream=downstream, state=state)
    assert evaluation.discharge == pytest.approx(1.2 + (1.8 - 0.9))
    assert evaluation.upstream_stage == pytest.approx(1.8)
    assert evaluation.downstream_stage == pytest.approx(0.9)

    refreshed = boundary.update_state(evaluation, state)
    assert refreshed.discharge == pytest.approx(evaluation.discharge)
    assert refreshed.metadata["last_discharge"] == pytest.approx(evaluation.discharge)
    assert refreshed is not state


def test_passive_gate_boundary_rating() -> None:
    gate = PassiveGateBoundary(
        boundary_id="gate1",
        face_index=0,
        upstream_node="up",
        downstream_node="down",
        width=4.0,
        invert_elevation=0.0,
        discharge_coeff=0.6,
        default_opening=0.5,
    )

    upstream = NodeState(stage=2.5)
    downstream = NodeState(stage=2.0)
    state = InternalBoundaryState(opening=0.5)

    evaluation = gate.evaluate(time=0.0, upstream=upstream, downstream=downstream, state=state)
    area = gate.width * 0.5
    head = upstream.stage - downstream.stage
    expected = gate.discharge_coeff * area * math.sqrt(2.0 * GRAVITY * head)
    assert evaluation.discharge == pytest.approx(expected, rel=1e-6)

    refreshed = gate.update_state(evaluation, state)
    assert refreshed.metadata["last_discharge"] == pytest.approx(expected, rel=1e-6)
    assert refreshed.opening == pytest.approx(0.5)


def test_fixed_speed_pump_boundary_operating_window() -> None:
    pump = FixedSpeedPumpBoundary(
        boundary_id="pump1",
        face_index=0,
        upstream_node="sump",
        downstream_node="delivery",
        rated_discharge=3.2,
        rated_head=1.5,
        shutoff_head=2.0,
        efficiency_curve=((0.0, 0.55), (3.2, 0.78), (5.0, 0.75)),
        control_mode="schedule",
        control_schedule=((0.0, "on"), (20.0, "off")),
    )
    upstream = NodeState(stage=10.0)
    downstream = NodeState(stage=11.0)
    state = InternalBoundaryState(pump_state="on")

    evaluation = pump.evaluate(time=0.0, upstream=upstream, downstream=downstream, state=state)
    assert evaluation.discharge == pytest.approx(3.2)
    assert evaluation.metadata["pump_on"] is True
    assert evaluation.metadata["efficiency"] == pytest.approx(0.78, rel=1e-6)

    refreshed = pump.update_state(evaluation, state)
    assert refreshed.discharge == pytest.approx(3.2)
    assert refreshed.pump_state == "on"
    assert refreshed.metadata["required_head"] == pytest.approx(1.0)
    assert refreshed.metadata["efficiency"] == pytest.approx(0.78, rel=1e-6)

    # 超过可用扬程时停泵
    downstream_high = NodeState(stage=12.0)
    eval_high = pump.evaluate(time=10.0, upstream=upstream, downstream=downstream_high, state=refreshed)
    assert eval_high.discharge == pytest.approx(0.0)
    assert eval_high.metadata["pump_on"] is True

    eval_off = pump.evaluate(time=25.0, upstream=upstream, downstream=downstream, state=refreshed)
    assert eval_off.metadata["pump_on"] is False
    assert eval_off.discharge == pytest.approx(0.0)
    refreshed_off = pump.update_state(eval_off, refreshed)
    assert refreshed_off.pump_state == "off"


def test_fixed_speed_pump_level_control() -> None:
    pump = FixedSpeedPumpBoundary(
        boundary_id="pump2",
        face_index=0,
        upstream_node="sump",
        downstream_node="delivery",
        rated_discharge=2.5,
        rated_head=1.0,
        control_mode="level",
        control_levels=(2.4, 2.0, "downstream"),
    )
    upstream = NodeState(stage=2.1)
    downstream = NodeState(stage=2.3)
    state = InternalBoundaryState(pump_state="off")

    eval_low = pump.evaluate(time=0.0, upstream=upstream, downstream=downstream, state=state)
    assert eval_low.metadata["pump_on"] is False

    downstream_high = NodeState(stage=2.5)
    eval_high = pump.evaluate(time=10.0, upstream=upstream, downstream=downstream_high, state=state)
    assert eval_high.metadata["pump_on"] is True
def test_hll_solver_applies_gate_boundary() -> None:
    sections = [
        CrossSection(0.0, 0.0, bottom_width=4.0, mannings_n=0.015),
        CrossSection(50.0, -0.05, bottom_width=4.0, mannings_n=0.015),
        CrossSection(100.0, -0.10, bottom_width=4.0, mannings_n=0.015),
        CrossSection(150.0, -0.15, bottom_width=4.0, mannings_n=0.015),
    ]
    profile = ChannelProfile(sections)
    gate = PassiveGateBoundary(
        boundary_id="gate-mid",
        face_index=1,
        upstream_node="sec1",
        downstream_node="sec2",
        width=4.0,
        invert_elevation=-0.05,
        discharge_coeff=0.6,
        default_opening=0.5,
    )

    initial_depths = [2.7, 2.7, 2.0, 2.0]
    up_stage = sections[1].bed_elevation + initial_depths[1]
    down_stage = sections[2].bed_elevation + initial_depths[2]
    head = max(0.0, up_stage - max(down_stage, gate.invert_elevation))
    expected_q = gate.discharge_coeff * gate.width * gate.default_opening * math.sqrt(2.0 * GRAVITY * head)

    solver = HLLMusclSolver()
    initial_discharges = [expected_q] * len(initial_depths)

    upstream_stage = lambda t, q: 2.7  # noqa: E731
    downstream_stage = lambda t, q: 2.0  # noqa: E731
    upstream_discharge = lambda t: expected_q * 1.2  # noqa: E731

    gate_state = InternalBoundaryState(opening=0.5)
    states_map = {"gate-mid": gate_state}

    results = solver.solve(
        profile=profile,
        total_time=0.2,
        dt=0.2,
        initial_depths=initial_depths,
        initial_discharges=initial_discharges,
        upstream_discharge=upstream_discharge,
        upstream_stage=upstream_stage,
        downstream_stage=downstream_stage,
        internal_boundaries=[gate],
        boundary_states=states_map,
    )

    assert states_map["gate-mid"] is not gate_state
    last_state = states_map["gate-mid"]
    opening = last_state.metadata["opening"]
    head_last = last_state.metadata["head"]
    rated_q = gate.discharge_coeff * gate.width * opening * math.sqrt(2.0 * GRAVITY * head_last)

    assert head_last > 0.0
    assert last_state.metadata["last_discharge"] == pytest.approx(rated_q, rel=1e-5)
    assert rated_q < upstream_discharge(0.0)


def test_preissmann_solver_handles_internal_boundaries() -> None:
    sections = [
        CrossSection(0.0, 0.0, bottom_width=5.0, mannings_n=0.02),
        CrossSection(40.0, -0.02, bottom_width=5.0, mannings_n=0.02),
        CrossSection(80.0, -0.04, bottom_width=5.0, mannings_n=0.02),
        CrossSection(120.0, -0.05, bottom_width=5.0, mannings_n=0.02),
        CrossSection(160.0, -0.06, bottom_width=5.0, mannings_n=0.02),
    ]
    profile = ChannelProfile(sections)

    gate = PassiveGateBoundary(
        boundary_id="gate-mid",
        face_index=1,
        upstream_node="sec1",
        downstream_node="sec2",
        width=5.0,
        invert_elevation=-0.02,
        discharge_coeff=0.6,
        default_opening=0.4,
    )
    pump = FixedSpeedPumpBoundary(
        boundary_id="pump-out",
        face_index=3,
        upstream_node="sec3",
        downstream_node="sec4",
        rated_discharge=4.0,
        rated_head=1.0,
        shutoff_head=1.4,
        min_discharge=2.5,
        efficiency_curve=((0.0, 0.5), (4.0, 0.8), (5.5, 0.75)),
    )

    initial_depths = [2.2, 2.3, 2.4, 2.5, 2.6]
    initial_discharges = [3.5, 3.5, 3.5, 3.5, 3.5]

    upstream_discharge = lambda t: 3.5  # noqa: E731
    upstream_stage = lambda t, q: 2.0  # noqa: E731
    downstream_stage = lambda t, q: 2.6  # noqa: E731

    states_map = {
        "gate-mid": InternalBoundaryState(opening=0.4),
        "pump-out": InternalBoundaryState(pump_state="on"),
    }

    solver = PreissmannImplicitSolver()
    results = solver.solve(
        profile=profile,
        total_time=120.0,
        dt=5.0,
        initial_depths=initial_depths,
        initial_discharges=initial_discharges,
        upstream_discharge=upstream_discharge,
        upstream_stage=upstream_stage,
        downstream_stage=downstream_stage,
        internal_boundaries=[gate, pump],
        boundary_states=states_map,
    )

    assert results.times.shape[0] > 1
    gate_state = states_map["gate-mid"]
    pump_state = states_map["pump-out"]
    assert "last_discharge" in gate_state.metadata
    assert "efficiency" in pump_state.metadata
    assert pump_state.metadata.get("energy_kwh", 0.0) > 0.0
    assert "history" in pump_state.metadata
    assert pump_state.pump_state in {"on", "off"}
