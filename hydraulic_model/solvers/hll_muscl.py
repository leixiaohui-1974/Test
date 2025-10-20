from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from hydraulic_model.geometry import ChannelProfile
from hydraulic_model.hydraulics import GRAVITY, manning_friction_slope
from hydraulic_model.unsteady import UnsteadyResults

from .internal_boundary import InternalBoundaryState, NodeState

if TYPE_CHECKING:
    from hydraulic_model.network import HydraulicNetwork
    from .internal_boundary import InternalBoundary, InternalBoundaryState

from .base import SolverConfig


def _minmod(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    if a == 0.0 or b == 0.0:
        return 0.0
    if np.signbit(a) != np.signbit(b):
        return 0.0
    return np.sign(a) * min(abs(a), abs(b))


def _accumulate_boundary_energy(
    boundary_id: str,
    evaluation,
    time_now: float,
    dt: float,
    boundary_states_map: dict[str, "InternalBoundaryState"],
    boundary_states: MutableMapping[str, "InternalBoundaryState"] | None,
) -> None:
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
    state_obj = boundary_states_map[boundary_id]
    prev_energy = float(state_obj.metadata.get("energy_kwh", 0.0))
    state_obj.metadata["energy_kwh"] = prev_energy + energy_kwh
    history = state_obj.metadata.setdefault("history", [])
    if isinstance(history, list):
        sample = {
            "time": time_now,
            "discharge": float(evaluation.discharge),
            "upstream_stage": float(evaluation.upstream_stage or 0.0),
            "downstream_stage": float(evaluation.downstream_stage or 0.0),
            "pump_on": bool(metadata.get("pump_on", True)),
            "efficiency": float(metadata.get("efficiency", 0.0)),
            "input_power_kw": power_val,
            "hydraulic_power_kw": float(metadata.get("hydraulic_power_kw", 0.0)),
            "energy_kwh": prev_energy + energy_kwh,
        }
        history.append(sample)
        state_obj.metadata["history"] = history
    boundary_states_map[boundary_id] = state_obj
    if boundary_states is not None:
        boundary_states[boundary_id] = state_obj


@dataclass
class HLLMusclSolver:
    """Finite-volume solver with MUSCL reconstruction and HLL flux."""

    config: SolverConfig = SolverConfig(
        name="hll_muscl",
        description="Finite-volume MUSCL-HLL solver with SSP-RK3 time stepping",
        is_implicit=False,
        citation="Toro, 2001; MUSCL-HLL shallow water schemes",
    )

    min_depth: float = 1e-4
    max_depth_limit: float = 50.0
    max_flux: float = 1e4

    def solve(
        self,
        profile: ChannelProfile,
        total_time: float,
        *,
        dt: float | None,
        initial_depths: Sequence[float],
        initial_discharges: Sequence[float],
        upstream_discharge: Callable[[float], float],
        upstream_stage: Callable[[float, float], float],
        downstream_stage: Callable[[float, float], float],
        internal_boundaries: Sequence["InternalBoundary"] | None = None,
        network: "HydraulicNetwork" | None = None,
        boundary_states: MutableMapping[str, "InternalBoundaryState"] | None = None,
    ) -> UnsteadyResults:
        if dt is None:
            raise ValueError("HLL-MUSCL solver需要显式给定时间步长 dt。")
        dt = float(dt)

        sections = profile.sections
        nsec = len(sections)
        if nsec < 3:
            raise ValueError("HLL-MUSCL solver 至少需要三个断面。")

        boundaries = list(internal_boundaries or [])
        boundary_states_map: dict[str, InternalBoundaryState] = {}
        if boundary_states is not None:
            boundary_states_map.update(boundary_states)

        face_boundaries: dict[int, list["InternalBoundary"]] = {}
        for boundary in boundaries:
            face_idx = boundary.face_index
            if face_idx < 0 or face_idx >= nsec - 1:
                raise ValueError(
                    f"Internal boundary {boundary.boundary_id} face_index={face_idx} 超出通量面范围 0..{nsec - 2}"
                )
            face_boundaries.setdefault(face_idx, []).append(boundary)
            state_obj = boundary_states_map.setdefault(boundary.boundary_id, InternalBoundaryState())
            if boundary_states is not None and boundary.boundary_id not in boundary_states:
                boundary_states[boundary.boundary_id] = state_obj

        bed = np.array([sec.bed_elevation for sec in sections], dtype=float)
        stations = profile.stations
        dx_face = np.diff(stations)
        cell_dx = np.zeros(nsec)
        cell_dx[0] = dx_face[0]
        cell_dx[-1] = dx_face[-1]
        for i in range(1, nsec - 1):
            cell_dx[i] = 0.5 * (dx_face[i - 1] + dx_face[i])

        # state variables (area and discharge)
        area = np.array([sec.area(depth) for sec, depth in zip(sections, initial_depths)], dtype=float)
        discharge = np.array(initial_discharges, dtype=float)
        depth = np.array(initial_depths, dtype=float)

        times = np.arange(0.0, total_time + dt, dt)
        depth_hist = np.zeros((len(times), nsec), dtype=float)
        discharge_hist = np.zeros((len(times), nsec), dtype=float)
        depth_hist[0] = depth
        discharge_hist[0] = discharge

        min_area = 1e-6
        max_areas = []
        for sec in sections:
            if sec.max_depth is not None:
                depth_lim = sec.max_depth + self.max_depth_limit
            else:
                depth_lim = self.max_depth_limit
            max_val = sec.area(depth_lim)
            max_areas.append(max(max_val, 1e4))
        max_areas = np.array(max_areas, dtype=float)

        h_min = self.min_depth

        def enforce_state(a: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a = np.where(~np.isfinite(a), min_area, a)
            q = np.where(~np.isfinite(q), 0.0, q)
            a = np.clip(a, min_area, max_areas)
            depths = np.array(
                [sec.depth_from_area(val) for sec, val in zip(sections, a)],
                dtype=float,
            )
            depths = np.clip(depths, 0.0, self.max_depth_limit)
            q = np.clip(q, -self.max_flux, self.max_flux)
            return a, q, depths

        def compute_rhs(a: np.ndarray, q: np.ndarray, time_now: float) -> tuple[np.ndarray, np.ndarray]:
            a, q, depths = enforce_state(a, q)

            top_widths = np.array(
                [sec.top_width(depth) for sec, depth in zip(sections, depths)], dtype=float
            )
            top_widths = np.maximum(top_widths, 1e-3)

            # MUSCL slopes for area and discharge
            slope_a = np.zeros_like(a)
            slope_q = np.zeros_like(q)
            for i in range(1, nsec - 1):
                dxm = cell_dx[i]
                left = (a[i] - a[i - 1]) / max(cell_dx[i - 1], 1e-6)
                right = (a[i + 1] - a[i]) / max(cell_dx[i + 1], 1e-6)
                slope_a[i] = _minmod(left, right) * dxm * 0.5

                left_q = (q[i] - q[i - 1]) / max(cell_dx[i - 1], 1e-6)
                right_q = (q[i + 1] - q[i]) / max(cell_dx[i + 1], 1e-6)
                slope_q[i] = _minmod(left_q, right_q) * dxm * 0.5

            # reconstruct states at interfaces
            a_left = np.zeros(nsec - 1)
            a_right = np.zeros(nsec - 1)
            q_left = np.zeros(nsec - 1)
            q_right = np.zeros(nsec - 1)

            for i in range(nsec - 1):
                a_left[i] = np.clip(a[i] + slope_a[i], min_area, max_areas[i])
                a_right[i] = np.clip(a[i + 1] - slope_a[i + 1], min_area, max_areas[i + 1])
                q_left[i] = np.clip(q[i] + slope_q[i], -self.max_flux, self.max_flux)
                q_right[i] = np.clip(q[i + 1] - slope_q[i + 1], -self.max_flux, self.max_flux)

            flux_a = np.zeros(nsec - 1)
            flux_q = np.zeros(nsec - 1)

            for i in range(nsec - 1):
                area_L = a_left[i]
                area_R = a_right[i]
                depth_L = sections[i].depth_from_area(area_L)
                depth_R = sections[i + 1].depth_from_area(area_R)
                qL = q_left[i]
                qR = q_right[i]

                if depth_L <= h_min and depth_R <= h_min:
                    flux_a[i] = 0.0
                    flux_q[i] = 0.0
                    continue

                B_L = max(sections[i].top_width(depth_L), 1e-3)
                B_R = max(sections[i + 1].top_width(depth_R), 1e-3)
                uL = qL / area_L
                uR = qR / area_R
                cL = np.sqrt(GRAVITY * area_L / B_L)
                cR = np.sqrt(GRAVITY * area_R / B_R)

                SL = min(uL - cL, uR - cR)
                SR = max(uL + cL, uR + cR)

                FL = np.array(
                    [
                        qL,
                        qL * uL + 0.5 * GRAVITY * depth_L * depth_L * B_L,
                    ]
                )
                FR = np.array(
                    [
                        qR,
                        qR * uR + 0.5 * GRAVITY * depth_R * depth_R * B_R,
                    ]
                )
                UL = np.array([area_L, qL])
                UR = np.array([area_R, qR])

                if SL >= 0.0:
                    flux = FL
                elif SR <= 0.0:
                    flux = FR
                else:
                    denom = SR - SL
                    if abs(denom) < 1e-8:
                        flux = 0.5 * (FL + FR)
                    else:
                        flux = (SR * FL - SL * FR + SL * SR * (UR - UL)) / denom

                flux = np.where(np.isfinite(flux), flux, 0.0)
                flux_a[i] = flux[0]
                flux_q[i] = flux[1]

            if face_boundaries:
                for face_idx, boundary_list in face_boundaries.items():
                    upstream_state = NodeState(
                        stage=sections[face_idx].water_surface_elevation(depths[face_idx]),
                        discharge=q[face_idx],
                    )
                    downstream_state = NodeState(
                        stage=sections[face_idx + 1].water_surface_elevation(depths[face_idx + 1]),
                        discharge=q[face_idx + 1],
                    )
                    for boundary in boundary_list:
                        state_obj = boundary_states_map.setdefault(
                            boundary.boundary_id, InternalBoundaryState()
                        )
                        evaluation = boundary.evaluate(time_now, upstream_state, downstream_state, state_obj)
                        flux_q[face_idx] = evaluation.discharge
                        new_state = boundary.update_state(evaluation, state_obj)
                        boundary_states_map[boundary.boundary_id] = new_state
                        if boundary_states is not None:
                            boundary_states[boundary.boundary_id] = new_state
                        _accumulate_boundary_energy(
                            boundary.boundary_id,
                            evaluation,
                            time_now,
                            dt,
                            boundary_states_map,
                            boundary_states,
                        )

            dadt = np.zeros_like(a)
            dqdt = np.zeros_like(q)

            for i in range(1, nsec - 1):
                dadt[i] = -(flux_a[i] - flux_a[i - 1]) / max(cell_dx[i], 1e-6)
                dqdt[i] = -(flux_q[i] - flux_q[i - 1]) / max(cell_dx[i], 1e-6)

            # boundary conditions via hydrograph/stage
            q_up = upstream_discharge(time_now)
            depth_up = max(upstream_stage(time_now, q_up), 0.0)
            area_up = sections[0].area(depth_up)
            dadt[0] = (area_up - a[0]) / dt
            dqdt[0] = (q_up - q[0]) / dt

            q_dn_guess = q[nsec - 2]
            depth_dn = max(downstream_stage(time_now, q_dn_guess), 0.0)
            area_dn = sections[-1].area(depth_dn)
            dadt[-1] = (area_dn - a[-1]) / dt
            dqdt[-1] = (q_dn_guess - q[-1]) / dt

            # source terms (bed slope + friction)
            bed_slope = np.zeros_like(a)
            bed_slope[0] = (bed[0] - bed[1]) / max(dx_face[0], 1e-6)
            bed_slope[-1] = (bed[-2] - bed[-1]) / max(dx_face[-1], 1e-6)
            if nsec > 2:
                denom = np.maximum(dx_face[1:], 1e-6)
                bed_slope[1:-1] = (bed[1:-1] - bed[2:]) / denom

            for i in range(1, nsec - 1):
                if depths[i] <= h_min:
                    continue
                A = a[i]
                Q = q[i]
                R = sections[i].hydraulic_radius(depths[i])
                if R <= 0.0:
                    continue
                Sf = manning_friction_slope(sections[i], depths[i], Q)
                dqdt[i] += GRAVITY * A * (bed_slope[i] - Sf)

            dadt = np.where(np.isfinite(dadt), dadt, 0.0)
            dqdt = np.where(np.isfinite(dqdt), dqdt, 0.0)

            return dadt, dqdt

        def SSP_RK3_step(a: np.ndarray, q: np.ndarray, time_now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            da1, dq1 = compute_rhs(a, q, time_now)
            a1 = a + dt * da1
            q1 = q + dt * dq1
            a1, q1, depths1 = enforce_state(a1, q1)

            da2, dq2 = compute_rhs(a1, q1, time_now + dt)
            a2 = 0.75 * a + 0.25 * (a1 + dt * da2)
            q2 = 0.75 * q + 0.25 * (q1 + dt * dq2)
            a2, q2, depths2 = enforce_state(a2, q2)

            da3, dq3 = compute_rhs(a2, q2, time_now + 0.5 * dt)
            a_new = (1.0 / 3.0) * a + (2.0 / 3.0) * (a2 + dt * da3)
            q_new = (1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dt * dq3)
            a_new, q_new, depths_new = enforce_state(a_new, q_new)
            return a_new, q_new, depths_new

        for idx in range(1, len(times)):
            t_now = times[idx - 1]
            area, discharge, depth = SSP_RK3_step(area, discharge, t_now)
            depth_hist[idx] = depth
            discharge_hist[idx] = discharge

        return UnsteadyResults(
            times=times,
            stations=profile.stations,
            depths=depth_hist,
            discharges=discharge_hist,
        )


