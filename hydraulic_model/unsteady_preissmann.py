from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .geometry import ChannelProfile
from .hydraulics import manning_discharge, manning_friction_slope
from .unsteady import UnsteadyResults

if TYPE_CHECKING:
    from hydraulic_model.solvers.internal_boundary import InternalBoundary, InternalBoundaryState, NodeState


def _accumulate_boundary_energy(
    boundary_id: str,
    evaluation,
    time_now: float,
    dt: float,
    boundary_states_map: dict[str, "InternalBoundaryState"],
    boundary_states: MutableMapping[str, "InternalBoundaryState"] | None,
) -> None:
    state = boundary_states_map[boundary_id]
    metadata = evaluation.metadata or {}
    if not isinstance(metadata, Mapping):
        return
    power_kw = metadata.get("input_power_kw")
    if power_kw is None:
        return
    try:
        power_val = float(power_kw)
    except (TypeError, ValueError):
        return
    energy_kwh = power_val * dt / 3600.0
    prev = float(state.metadata.get("energy_kwh", 0.0))
    state.metadata["energy_kwh"] = prev + energy_kwh
    history = state.metadata.setdefault("history", [])
    if isinstance(history, list):
        history.append(
            {
                "time": time_now,
                "discharge": float(evaluation.discharge),
                "upstream_stage": float(evaluation.upstream_stage or 0.0),
                "downstream_stage": float(evaluation.downstream_stage or 0.0),
                "pump_on": bool(metadata.get("pump_on", True)),
                "efficiency": float(metadata.get("efficiency", 0.0)),
                "input_power_kw": power_val,
                "hydraulic_power_kw": float(metadata.get("hydraulic_power_kw", 0.0)),
                "energy_kwh": prev + energy_kwh,
            }
        )
    boundary_states_map[boundary_id] = state
    if boundary_states is not None:
        boundary_states[boundary_id] = state


@dataclass
class PreissmannConfig:
    theta: float = 0.7
    max_iterations: int = 25
    convergence_tol: float = 1e-4
    min_depth: float = 1e-4
    relaxation: float = 0.05
    max_q_factor: float = 0.2
    discharge_relax: float = 0.3


class PreissmannSolver:
    """Implicit Preissmann scheme (Picard iterations) for 1D Saint-Venant equations."""

    def __init__(self, profile: ChannelProfile, config: Optional[PreissmannConfig] = None) -> None:
        self.profile = profile
        self.config = config or PreissmannConfig()
        self._stations = profile.stations
        self._dx = np.diff(self._stations)
        self._widths = np.array([section.bottom_width for section in profile.sections], dtype=float)
        self._bed_elev = np.array(
            [section.bed_elevation for section in profile.sections], dtype=float
        )

    def solve(
        self,
        total_time: float,
        *,
        dt: float,
        initial_depths: Optional[Sequence[float]] = None,
        initial_discharges: Optional[Sequence[float]] = None,
        upstream_discharge: Callable[[float], float],
        upstream_stage: Callable[[float, float], float],
        downstream_stage: Callable[[float, float], float],
        internal_boundaries: Sequence["InternalBoundary"] | None = None,
        boundary_states: MutableMapping[str, "InternalBoundaryState"] | None = None,
    ) -> UnsteadyResults:
        theta = self.config.theta
        min_depth = self.config.min_depth
        relax = self.config.relaxation
        q_factor = self.config.max_q_factor
        discharge_relax = self.config.discharge_relax

        num_sections = len(self.profile.sections)
        times = np.arange(0.0, total_time + dt, dt)

        boundaries = list(internal_boundaries or [])
        boundary_states_map: dict[str, "InternalBoundaryState"] = {}
        if boundary_states is not None:
            boundary_states_map.update(boundary_states)
        face_boundaries: dict[int, list["InternalBoundary"]] = {}
        if boundaries:
            from hydraulic_model.solvers.internal_boundary import InternalBoundaryState  # Local import to avoid cycles

        for boundary in boundaries:
            face_idx = boundary.face_index
            if face_idx < 0 or face_idx >= num_sections - 1:
                raise ValueError(
                    f"Internal boundary {boundary.boundary_id} face_index={face_idx} 超出可用范围 0..{num_sections - 2}"
                )
            if face_idx == 0 or face_idx == num_sections - 2:
                # allow but warn? we can support but still interior. upstream index 0 corresponds to first interior face.
                pass
            face_boundaries.setdefault(face_idx, []).append(boundary)
            state_obj = boundary_states_map.setdefault(boundary.boundary_id, InternalBoundaryState())
            if boundary_states is not None and boundary.boundary_id not in boundary_states:
                boundary_states[boundary.boundary_id] = state_obj

        if initial_depths is None:
            depths = np.full(num_sections, 1.0, dtype=float)
            reference_depths = depths.copy()
        else:
            depths = np.array(initial_depths, dtype=float, copy=True)
            reference_depths = depths.copy()

        if initial_discharges is None:
            discharges = np.zeros(num_sections, dtype=float)
        else:
            discharges = np.array(initial_discharges, dtype=float, copy=True)

        depth_history = np.zeros((len(times), num_sections), dtype=float)
        discharge_history = np.zeros((len(times), num_sections), dtype=float)
        depth_history[0] = depths
        discharge_history[0] = discharges

        sections = self.profile.sections

        for step in range(1, len(times)):
            t_curr = times[step - 1]
            t_next = times[step]

            q_up = max(upstream_discharge(t_next), 0.0)
            discharges[0] = q_up
            depths[0] = max(upstream_stage(t_next, q_up), min_depth)
            q_down = discharges[-2]
            depths[-1] = max(downstream_stage(t_next, q_down), min_depth)
            discharges[-1] = discharges[-2]

            old_depths = depth_history[step - 1].copy()
            old_discharges = discharge_history[step - 1].copy()

            new_depths = depths.copy()
            new_discharges = discharges.copy()

            for iteration in range(self.config.max_iterations):
                prev_depths = new_depths.copy()
                prev_discharges = new_discharges.copy()

                # apply boundaries at iteration start
                q_up = max(upstream_discharge(t_next), 0.0)
                new_discharges[0] = q_up
                new_depths[0] = max(upstream_stage(t_next, q_up), min_depth)
                q_down = new_discharges[-2]
                new_depths[-1] = max(downstream_stage(t_next, q_down), min_depth)
                new_discharges[-1] = new_discharges[-2]

                for i in range(1, num_sections - 1):
                    dx = self._dx[i - 1]
                    section = sections[i]

                    depth_avg = theta * prev_depths[i] + (1.0 - theta) * old_depths[i]
                    depth_avg = max(depth_avg, min_depth)
                    area_avg = section.area(depth_avg)
                    hydraulic_radius = section.hydraulic_radius(depth_avg)
                    if hydraulic_radius <= 0.0:
                        hydraulic_radius = min_depth
                    bed_slope = (self._bed_elev[i - 1] - self._bed_elev[i]) / dx

                    h_i_new = self._bed_elev[i] + prev_depths[i]
                    h_im1_new = self._bed_elev[i - 1] + prev_depths[i - 1]
                    h_i_old = self._bed_elev[i] + old_depths[i]
                    h_im1_old = self._bed_elev[i - 1] + old_depths[i - 1]

                    dHdx = (
                        theta * (h_i_new - h_im1_new) + (1.0 - theta) * (h_i_old - h_im1_old)
                    ) / dx

                    q_avg = theta * prev_discharges[i] + (1.0 - theta) * old_discharges[i]
                    if depth_avg <= 0.05:
                        friction_slope = 0.0
                    else:
                        friction_slope = manning_friction_slope(section, depth_avg, q_avg)
                        friction_slope = float(np.clip(friction_slope, 0.0, 5.0))

                    sign_q = np.sign(q_avg) if q_avg != 0.0 else 1.0
                    momentum_term = float(np.clip(dHdx + sign_q * friction_slope, -0.02, 0.02))

                    target_q = old_discharges[i] - dt * 9.81 * area_avg * momentum_term
                    max_ref = max(abs(prev_discharges[i]), abs(old_discharges[i]), 1.0)
                    limit = q_factor * max_ref
                    target_q = float(np.clip(target_q, -limit, limit))
                    new_q = (1.0 - relax) * prev_discharges[i] + relax * target_q
                    if not np.isfinite(new_q):
                        new_q = prev_discharges[i]
                    new_q = float(np.clip(new_q, -500.0, 500.0))

                    new_discharges[i] = new_q

                for i in range(1, num_sections - 1):
                    dx = self._dx[i - 1]
                    section = sections[i]
                    area_old = section.area(old_depths[i])
                    flux_new = theta * (
                        new_discharges[i] - new_discharges[i - 1]
                    ) + (1.0 - theta) * (old_discharges[i] - old_discharges[i - 1])
                    area_new = area_old - (dt / dx) * flux_new
                    min_area = min_depth * self._widths[i]
                    if area_new < min_area:
                        area_new = min_area
                    depth_new = max(area_new / self._widths[i], min_depth)
                    if not np.isfinite(depth_new):
                        depth_new = prev_depths[i]
                    depth_new = float(np.clip(depth_new, prev_depths[i] - 0.5, prev_depths[i] + 0.5))
                    depth_cap_low = reference_depths[i] - 2.0
                    depth_cap_high = reference_depths[i] + 2.0
                    depth_new = float(np.clip(depth_new, depth_cap_low, depth_cap_high))
                    new_depths[i] = depth_new

                for i in range(1, num_sections - 1):
                    slope_local = abs(self._bed_elev[i - 1] - self._bed_elev[i]) / self._dx[i - 1]
                    slope_local = max(slope_local, 1e-4)
                    rating_q = manning_discharge(sections[i], new_depths[i], slope_local)
                    new_discharges[i] = (1.0 - discharge_relax) * new_discharges[i] + discharge_relax * rating_q

                if face_boundaries:
                    from hydraulic_model.solvers.internal_boundary import NodeState  # Local import to avoid cycles

                    for face_idx, boundary_list in face_boundaries.items():
                        upstream_stage_val = self._bed_elev[face_idx] + new_depths[face_idx]
                        downstream_stage_val = self._bed_elev[face_idx + 1] + new_depths[face_idx + 1]
                        upstream_state = NodeState(
                            stage=upstream_stage_val,
                            discharge=new_discharges[face_idx],
                        )
                        downstream_state = NodeState(
                            stage=downstream_stage_val,
                            discharge=new_discharges[face_idx + 1],
                        )
                        for boundary in boundary_list:
                            state_obj = boundary_states_map[boundary.boundary_id]
                            evaluation = boundary.evaluate(t_next, upstream_state, downstream_state, state_obj)
                            new_discharges[face_idx] = evaluation.discharge
                            upstream_state.discharge = evaluation.discharge
                            downstream_state.discharge = evaluation.discharge
                            new_state = boundary.update_state(evaluation, state_obj)
                            boundary_states_map[boundary.boundary_id] = new_state
                            if boundary_states is not None:
                                boundary_states[boundary.boundary_id] = new_state
                            _accumulate_boundary_energy(
                                boundary.boundary_id,
                                evaluation,
                                t_next,
                                dt,
                                boundary_states_map,
                                boundary_states,
                            )

                change_depth = np.max(np.abs(new_depths - prev_depths))
                change_q = np.max(np.abs(new_discharges - prev_discharges))
                if max(change_depth, change_q) < self.config.convergence_tol:
                    break

            depths = new_depths
            discharges = new_discharges

            depth_history[step] = depths
            discharge_history[step] = discharges

        return UnsteadyResults(
            times=times,
            stations=self._stations,
            depths=depth_history,
            discharges=discharge_history,
        )




















