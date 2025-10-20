from __future__ import annotations

"""Utilities for constructing internal boundary objects from configuration files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from hydraulic_model.solvers import (
    FixedSpeedPumpBoundary,
    InternalBoundary,
    InternalBoundaryState,
    PassiveGateBoundary,
)


class InternalBoundaryConfigError(ValueError):
    """Raised when an internal boundary definition is invalid."""


@dataclass(frozen=True)
class LoadedBoundaries:
    """Bundle containing boundary objects and their associated states."""

    boundaries: list[InternalBoundary]
    states: dict[str, InternalBoundaryState]


def _ensure_sequence(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        if "boundaries" in data:
            return _ensure_sequence(data["boundaries"])
        raise InternalBoundaryConfigError("配置文件需要包含 'boundaries' 数组或直接提供边界列表。")
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [item for item in data]  # type: ignore[arg-type]
    raise InternalBoundaryConfigError("无法解析内边界配置，需提供列表或包含 'boundaries' 的对象。")


def _parse_state(defn: Mapping[str, Any]) -> InternalBoundaryState:
    opening = defn.get("opening")
    discharge = defn.get("discharge", 0.0)
    pump_state = defn.get("pump_state")
    metadata_raw = defn.get("metadata") or {}
    if not isinstance(metadata_raw, Mapping):
        raise InternalBoundaryConfigError("state.metadata 必须为对象。")
    metadata: MutableMapping[str, Any] = dict(metadata_raw)
    return InternalBoundaryState(
        discharge=float(discharge),
        opening=None if opening is None else float(opening),
        pump_state=None if pump_state is None else str(pump_state),
        metadata=metadata,
    )


def _build_passive_gate(defn: Mapping[str, Any]) -> tuple[PassiveGateBoundary, InternalBoundaryState]:
    required = ["id", "face_index", "width", "invert_elevation"]
    missing = [field for field in required if field not in defn]
    if missing:
        raise InternalBoundaryConfigError(f"缺少闸门字段: {', '.join(missing)}")

    boundary = PassiveGateBoundary(
        boundary_id=str(defn["id"]),
        face_index=int(defn["face_index"]),
        upstream_node=str(defn.get("upstream_node", "")),
        downstream_node=str(defn.get("downstream_node", "")),
        width=float(defn["width"]),
        invert_elevation=float(defn["invert_elevation"]),
        discharge_coeff=float(defn.get("discharge_coeff", 0.6)),
        default_opening=float(defn.get("default_opening", 0.0)),
        min_opening=float(defn.get("min_opening", 0.0)),
        max_opening=float(defn.get("max_opening", 5.0)),
    )

    state_def = defn.get("state", {})
    if state_def and not isinstance(state_def, Mapping):
        raise InternalBoundaryConfigError("state 必须为对象。")
    state = _parse_state(state_def)
    return boundary, state


def _parse_efficiency_curve(data: Any) -> tuple[tuple[float, float], ...] | None:
    if data is None:
        return None
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise InternalBoundaryConfigError("efficiency_curve 必须为 [[流量, 效率], ...] 数组。")
    points: list[tuple[float, float]] = []
    for entry in data:
        if not isinstance(entry, Sequence) or len(entry) != 2:
            raise InternalBoundaryConfigError("efficiency_curve 中的每个元素都应为二元数组。")
        flow, eff = entry
        points.append((float(flow), float(eff)))
    if not points:
        return None
    return tuple(points)


def _parse_control(defn: Mapping[str, Any]) -> dict[str, Any]:
    control = defn.get("control")
    if control is None:
        return {}
    if not isinstance(control, Mapping):
        raise InternalBoundaryConfigError("control 必须为对象。")
    mode = str(control.get("mode", "schedule")).lower()
    info: dict[str, Any] = {"mode": mode}
    if mode == "schedule":
        entries = control.get("entries")
        if not isinstance(entries, Sequence):
            raise InternalBoundaryConfigError("control.entries 必须为数组。")
        schedule: list[tuple[float, str]] = []
        for entry in entries:
            if not isinstance(entry, Sequence) or len(entry) != 2:
                raise InternalBoundaryConfigError("control.entries 中的每个元素需为 [time, state]。")
            time_val, state_val = entry
            try:
                time_float = float(time_val)
            except (TypeError, ValueError) as exc:
                raise InternalBoundaryConfigError("control.entries 中的时间必须为数字。") from exc
            state_str = str(state_val).lower()
            if state_str not in {"on", "off"}:
                raise InternalBoundaryConfigError("control.entries 的状态仅支持 'on'/'off'。")
            schedule.append((time_float, state_str))
        schedule.sort(key=lambda item: item[0])
        info["schedule"] = tuple(schedule)
    elif mode == "level":
        try:
            on_level = float(control["on_level"])
            off_level = float(control.get("off_level", on_level - 0.1))
        except (KeyError, TypeError, ValueError) as exc:
            raise InternalBoundaryConfigError("level 控制需要 on_level/off_level 数值。") from exc
        sense = str(control.get("sense", "upstream")).lower()
        if sense not in {"upstream", "downstream"}:
            raise InternalBoundaryConfigError("level 控制的 sense 仅支持 upstream/downstream。")
        info["levels"] = (on_level, off_level, sense)
    elif mode == "cycle":
        try:
            on_duration = float(control["on_duration"])
            off_duration = float(control["off_duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InternalBoundaryConfigError("cycle 控制需要 on_duration/off_duration 数值。") from exc
        phase = float(control.get("phase", 0.0))
        info["cycle"] = (max(on_duration, 0.0), max(off_duration, 0.0), phase)
    else:
        raise InternalBoundaryConfigError(f"暂不支持的 control.mode: {mode}")
    return info


def _build_fixed_speed_pump(defn: Mapping[str, Any]) -> tuple[FixedSpeedPumpBoundary, InternalBoundaryState]:
    required = ["id", "face_index", "rated_discharge", "rated_head"]
    missing = [field for field in required if field not in defn]
    if missing:
        raise InternalBoundaryConfigError(f"缺少泵站字段: {', '.join(missing)}")

    efficiency_curve = _parse_efficiency_curve(defn.get("efficiency_curve"))
    control_info = _parse_control(defn)

    boundary = FixedSpeedPumpBoundary(
        boundary_id=str(defn["id"]),
        face_index=int(defn["face_index"]),
        upstream_node=str(defn.get("upstream_node", "")),
        downstream_node=str(defn.get("downstream_node", "")),
        rated_discharge=float(defn["rated_discharge"]),
        rated_head=float(defn["rated_head"]),
        shutoff_head=None if defn.get("shutoff_head") is None else float(defn["shutoff_head"]),
        min_discharge=float(defn.get("min_discharge", 0.0)),
        efficiency_curve=efficiency_curve,
        control_mode=control_info.get("mode"),
        control_schedule=control_info.get("schedule"),
        control_levels=control_info.get("levels"),
        control_cycle=control_info.get("cycle"),
    )

    state_def = defn.get("state", {})
    if state_def and not isinstance(state_def, Mapping):
        raise InternalBoundaryConfigError("state 必须为对象。")
    state = _parse_state(state_def)
    if state.pump_state is None:
        state.pump_state = "on"
    return boundary, state


def load_internal_boundaries(config_path: Path | str) -> LoadedBoundaries:
    """Read a JSON configuration describing internal boundaries."""

    path = Path(config_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise InternalBoundaryConfigError(f"无法读取内边界配置文件: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InternalBoundaryConfigError(f"配置 JSON 解析失败: {exc}") from exc

    entries = _ensure_sequence(data)
    boundaries: list[InternalBoundary] = []
    states: dict[str, InternalBoundaryState] = {}

    for item in entries:
        if not isinstance(item, Mapping):
            raise InternalBoundaryConfigError("内边界配置项必须为对象。")

        btype = str(item.get("type", "passive_gate")).lower()
        if btype == "passive_gate":
            boundary, state = _build_passive_gate(item)
        elif btype in {"fixed_speed_pump", "pump"}:
            boundary, state = _build_fixed_speed_pump(item)
        else:
            raise InternalBoundaryConfigError(f"暂不支持的内边界类型: {btype}")

        if boundary.boundary_id in states:
            raise InternalBoundaryConfigError(f"重复的边界 ID: {boundary.boundary_id}")

        boundaries.append(boundary)
        states[boundary.boundary_id] = state

    return LoadedBoundaries(boundaries=boundaries, states=states)
