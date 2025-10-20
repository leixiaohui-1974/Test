from __future__ import annotations

"""Pump-related internal boundary implementations."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence, Tuple

from .internal_boundary import BoundaryEvaluation, InternalBoundaryState, NodeState
from hydraulic_model.hydraulics import GRAVITY


WATER_DENSITY = 1000.0  # kg/m^3

@dataclass
class FixedSpeedPumpBoundary:
    """Simplified fixed-speed pump that supplies rated flow within its head limit."""

    boundary_id: str
    face_index: int
    upstream_node: str
    downstream_node: str
    rated_discharge: float
    rated_head: float
    shutoff_head: float | None = None
    min_discharge: float = 0.0
    efficiency_curve: tuple[tuple[float, float], ...] | None = None
    control_mode: str | None = None
    control_schedule: tuple[tuple[float, str], ...] | None = None
    control_levels: tuple[float, float, str] | None = None  # on_level, off_level, sense ('upstream'/'downstream')
    control_cycle: tuple[float, float, float] | None = None  # on_duration, off_duration, phase

    def _pump_enabled(self, state: InternalBoundaryState) -> bool:
        status = (state.pump_state or "on").lower()
        return status not in {"off", "stopped", "0", "false"}

    def _available_head(self) -> float:
        if self.shutoff_head is None:
            return max(self.rated_head, 0.0)
        return max(self.shutoff_head, 0.0)

    def _resolve_control(
        self,
        time: float,
        upstream: NodeState,
        downstream: NodeState,
        state: InternalBoundaryState,
    ) -> None:
        mode = (self.control_mode or "").lower()
        if mode == "schedule" and self.control_schedule:
            entries = self.control_schedule
            times = [item[0] for item in entries]
            idx = bisect_right(times, time) - 1
            if idx >= 0:
                state.pump_state = entries[idx][1]
        elif mode == "level" and self.control_levels:
            on_level, off_level, sense = self.control_levels
            stage = upstream.stage if sense == "upstream" else downstream.stage
            current = (state.pump_state or "off").lower()
            if stage >= on_level:
                state.pump_state = "on"
            elif stage <= off_level:
                state.pump_state = "off"
            else:
                state.pump_state = current
        elif mode == "cycle" and self.control_cycle:
            on_dur, off_dur, phase = self.control_cycle
            period = max(on_dur + off_dur, 1e-3)
            local_t = (time - phase) % period
            state.pump_state = "on" if local_t < on_dur else "off"

    def evaluate(
        self,
        time: float,
        upstream: NodeState,
        downstream: NodeState,
        state: InternalBoundaryState,
    ) -> BoundaryEvaluation:
        self._resolve_control(time, upstream, downstream, state)
        pump_on = self._pump_enabled(state)
        rated_head = max(self.rated_head, 0.0)
        available_head = self._available_head()
        head_capacity = min(available_head, rated_head)
        required_head = max(downstream.stage - upstream.stage, 0.0)

        discharge = 0.0
        if pump_on and required_head <= head_capacity:
            discharge = max(self.rated_discharge, self.min_discharge, 0.0)

        efficiency = None
        if self.efficiency_curve and discharge > 0.0:
            efficiency = _interpolate_efficiency(self.efficiency_curve, discharge)

        hydraulic_power_kw = 0.0
        input_power_kw = 0.0
        delivered_head = min(required_head, head_capacity) if pump_on else 0.0
        if discharge > 0.0 and delivered_head > 0.0:
            hydraulic_power_kw = WATER_DENSITY * GRAVITY * discharge * delivered_head / 1000.0
            if efficiency and efficiency > 0.0:
                input_power_kw = hydraulic_power_kw / efficiency
            else:
                input_power_kw = hydraulic_power_kw

        metadata: MutableMapping[str, float | bool] = {
            "pump_on": pump_on,
            "required_head": required_head,
            "available_head": available_head,
            "head_capacity": head_capacity,
            "delivered_head": delivered_head,
            "time": time,
        }
        if efficiency is not None:
            metadata["efficiency"] = efficiency
        metadata["hydraulic_power_kw"] = hydraulic_power_kw
        metadata["input_power_kw"] = input_power_kw
        return BoundaryEvaluation(
            discharge=discharge,
            upstream_stage=upstream.stage,
            downstream_stage=downstream.stage,
            metadata=metadata,
        )

    def update_state(
        self,
        evaluation: BoundaryEvaluation,
        state: InternalBoundaryState,
    ) -> InternalBoundaryState:
        metadata: Mapping[str, float | bool] | None = evaluation.metadata
        updated_meta: MutableMapping[str, float | bool] = dict(state.metadata)
        updated_meta["last_discharge"] = evaluation.discharge
        if metadata:
            updated_meta.update(
                {
                    k: v
                    for k, v in metadata.items()
                    if k in {"pump_on", "required_head", "head_capacity", "efficiency", "hydraulic_power_kw", "input_power_kw", "delivered_head"}
                }
            )

        pump_state = state.pump_state
        if metadata and "pump_on" in metadata:
            pump_state = "on" if metadata["pump_on"] else "off"

        return InternalBoundaryState(
            discharge=evaluation.discharge,
            opening=state.opening,
            pump_state=pump_state,
            metadata=updated_meta,
        )


def _interpolate_efficiency(
    curve: tuple[tuple[float, float], ...],
    discharge: float,
) -> float:
    points: Sequence[Tuple[float, float]] = tuple(sorted(curve, key=lambda item: item[0]))
    if not points:
        return 0.0

    if discharge <= points[0][0]:
        return float(points[0][1])
    if discharge >= points[-1][0]:
        return float(points[-1][1])

    xs = [p[0] for p in points]
    idx = bisect_left(xs, discharge)
    if idx == 0:
        return float(points[0][1])
    x0, y0 = points[idx - 1]
    x1, y1 = points[idx]
    if x1 == x0:
        return float(y1)
    ratio = (discharge - x0) / (x1 - x0)
    return float(y0 + ratio * (y1 - y0))
