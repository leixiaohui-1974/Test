from __future__ import annotations

from hydraulic_model.cli.common import (
    bed_elevations,
    build_common_parser,
    prepare_simulation_state,
    resolve_common_time_step,
    resolve_pulse,
)
from hydraulic_model.solvers import PreissmannImplicitSolver
from pathlib import Path

from hydraulic_model.workflows import (
    create_step_discharge,
    ensure_simulation_time,
    export_unsteady_diagnostics,
    load_internal_boundaries,
    run_unsteady_solver,
)


def main() -> None:
    parser = build_common_parser("Run the Preissmann implicit solver workflow.", "outputs/preissmann")
    parser.add_argument(
        "--internal-boundaries",
        type=Path,
        default=None,
        help="指向包含内边界定义的 JSON 文件。",
    )
    args = parser.parse_args()

    profile, upstream_stage_fn, downstream_stage_fn, steady_results = prepare_simulation_state(
        args.scenario, args.steady_discharge
    )

    resolved_total_time = ensure_simulation_time(profile, steady_results, args.total_time)
    dt_common = resolve_common_time_step(args.scenario, profile, steady_results, args.dt)
    dt_preissmann = min(dt_common, 5.0)
    pulse_amp = resolve_pulse(args.scenario, args.pulse_amplitude)
    base_flow = float(steady_results.discharges[0])
    upstream_discharge_fn = create_step_discharge(
        base_flow,
        step_time=args.step_time,
        pulse=pulse_amp,
    )

    scenario_dir = args.output_dir / args.scenario.replace(" ", "_")

    boundaries_arg = None
    boundary_states = None
    if args.internal_boundaries is not None:
        bundle = load_internal_boundaries(args.internal_boundaries)
        boundaries_arg = bundle.boundaries
        boundary_states = bundle.states

    results = run_unsteady_solver(
        PreissmannImplicitSolver(),
        profile,
        steady_results,
        total_time=resolved_total_time,
        dt=dt_preissmann,
        upstream_stage_fn=upstream_stage_fn,
        downstream_stage_fn=downstream_stage_fn,
        upstream_discharge=upstream_discharge_fn,
        internal_boundaries=boundaries_arg,
        boundary_states=boundary_states,
    )
    export_unsteady_diagnostics(
        results,
        bed_elevations(profile),
        scenario_dir,
        scenario=args.scenario,
        solver_name="preissmann",
        boundary_states=boundary_states or {},
    )


if __name__ == "__main__":
    main()
