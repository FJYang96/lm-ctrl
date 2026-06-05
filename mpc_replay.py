#!/usr/bin/env python3
"""
Replay MPC trajectory optimization from saved LLM constraint code (no LLM calls).

Use this to debug MPC constraints, the solver, and downstream tooling without
running the full llm_main.py feedback loop.

Prefer ./run_mpc_replay.sh (sources run_common.sh thread limits for IPOPT/MUMPS).

Examples:
    ./run_mpc_replay.sh results/example/sideflip.py
    python mpc_replay.py --constraint-file results/example/sideflip.py
    python mpc_replay.py --constraint-file results/example/sideflip.py --no-slack
    python mpc_replay.py --constraint-from-summary \\
        results/llm_iterations/.../summary_prompt_iter_2.txt \\
        --output-dir results/mpc_replay/sideflip_iter2
    python mpc_replay.py --constraint-file results/example/sideflip.py --skip-render
    python mpc_replay.py --constraint-file results/example/sideflip.py --solver-verbose
    MPC_SOLVER_VERBOSE=1 python mpc_replay.py --constraint-file results/example/sideflip.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import go2_config


def _load_constraint_file(path: Path) -> str:
    code = path.read_text(encoding="utf-8")
    if not code.strip():
        raise ValueError(f"Constraint file is empty: {path}")
    return code


def _extract_constraint_code_from_summary(path: Path) -> str:
    """Extract Python from a summary/scoring prompt log CONSTRAINT CODE section."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^\s*CONSTRAINT CODE\s*$\n(?:=+\s*\n)?",
        text,
    )
    if not match:
        raise ValueError(
            f"No 'CONSTRAINT CODE' section found in {path}. "
            "Use --constraint-file with a saved .py instead."
        )
    code = text[match.end() :].strip()
    # Drop trailing prompt sections if present (rare in summary files).
    for stop_marker in (
        "\n============================================================",
        "\n=== ",
    ):
        idx = code.find(stop_marker)
        if idx > 0:
            code = code[:idx].strip()
    if not code.startswith(("import ", "def configure_mpc")):
        raise ValueError(
            f"Extracted code from {path} does not look like MPC configuration Python."
        )
    return code


def _normalize_codegen_prefixed_code(code: str) -> str:
    """Strip '| ' prefixes from code embedded in codegen_prompt_iter_*.txt."""
    lines = code.splitlines()
    if not lines or not all(
        ln.startswith("| ") or ln.strip() == "|" or ln == "|" for ln in lines if ln.strip()
    ):
        return code
    return "\n".join(ln[2:] if ln.startswith("| ") else ln.lstrip("|") for ln in lines)


def _load_constraint_code(constraint_file: Path | None, summary_file: Path | None) -> str:
    if constraint_file is not None:
        return _load_constraint_file(constraint_file)
    assert summary_file is not None
    code = _extract_constraint_code_from_summary(summary_file)
    return _normalize_codegen_prefixed_code(code)


def _print_optimization_summary(result: dict[str, Any], output_dir: Path, iteration: int) -> None:
    metrics = result.get("optimization_metrics") or {}
    analysis = result.get("trajectory_analysis") or {}
    converged = result.get("success", False)

    print()
    print("=" * 60)
    print("MPC REPLAY RESULTS")
    print("=" * 60)
    print(f"Solver converged: {'yes' if converged else 'no'}")
    if metrics.get("error_message"):
        print(f"Error: {metrics['error_message']}")
    if metrics.get("inf_pr") is not None:
        print(f"Residuals: inf_pr={metrics['inf_pr']:.2e}, inf_du={metrics['inf_du']:.2e}")
    if metrics.get("solver_iterations") is not None:
        print(f"Solver iterations: {metrics['solver_iterations']}")
    if analysis and "error" not in analysis:
        print(
            f"Trajectory: duration={analysis.get('trajectory_duration', 0):.2f}s, "
            f"height_gain={analysis.get('height_gain', 0):.3f}m, "
            f"max_com_vel={analysis.get('max_com_velocity', 0):.2f}m/s"
        )
        print(
            f"  pitch total={analysis.get('total_pitch_rotation', 0):.2f}rad, "
            f"roll total={analysis.get('total_roll_rotation', 0):.2f}rad"
        )

    print()
    print(f"Output directory: {output_dir}")
    for name in (
        f"state_traj_iter_{iteration}.npy",
        f"grf_traj_iter_{iteration}.npy",
        f"joint_vel_traj_iter_{iteration}.npy",
        f"contact_sequence_iter_{iteration}.npy",
    ):
        p = output_dir / name
        if p.exists():
            print(f"  {p.name}")
    video = (
        output_dir / f"planned_traj_iter_{iteration}.mp4"
        if converged
        else output_dir / f"debug_trajectory_iter_{iteration}.mp4"
    )
    if video.exists():
        print(f"  {video.name}")


def run_replay(args: argparse.Namespace) -> int:
    from llm_integration import FeedbackPipeline
    from llm_integration.mpc import LLMTaskMPC
    from llm_integration.pipeline.optimization import solve_trajectory_optimization
    from llm_integration.pipeline.simulation import execute_simulation

    constraint_file = Path(args.constraint_file).resolve() if args.constraint_file else None
    summary_file = (
        Path(args.constraint_from_summary).resolve()
        if args.constraint_from_summary
        else None
    )

    try:
        code = _load_constraint_code(constraint_file, summary_file)
    except (OSError, ValueError) as e:
        print(f"Error loading constraint code: {e}")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    go2_config.CONSTRAINT_MODE = args.config
    use_slack = not args.no_slack
    solver_verbose = args.solver_verbose or (
        os.getenv("MPC_SOLVER_VERBOSE", "").lower() in ("1", "true", "yes")
    )
    go2_config.set_solver_verbose(solver_verbose)

    print("=" * 60)
    print("MPC Replay (no LLM)")
    print("=" * 60)
    if constraint_file:
        print(f"Constraint file: {constraint_file}")
    else:
        print(f"Summary file:    {summary_file}")
    print(f"Output dir:      {output_dir}")
    print(f"Iteration tag:   {args.iteration}")
    print(f"Slack:           {'disabled' if args.no_slack else 'enabled'}")
    print(f"Config mode:     {args.config}")
    print(f"Render:          {'no' if args.skip_render else 'yes'}")
    print(f"Solver verbose:  {'yes' if solver_verbose else 'no'}")
    print()

    pipeline = FeedbackPipeline(use_slack=use_slack)
    task_mpc = LLMTaskMPC(pipeline.kindyn_model, use_slack=use_slack)

    ok, err = pipeline.safe_executor.execute_mpc_configuration_code(code, task_mpc)
    if not ok:
        print(f"MPC configuration failed:\n{err}")
        return 1

    pipeline.current_task_mpc = task_mpc
    config_summary = task_mpc.get_configuration_summary()
    task_name = config_summary["task_name"]
    print(
        f"Configured: {task_name}, duration={config_summary['duration']:.2f}s, "
        f"dt={config_summary['time_step']:.3f}s, "
        f"constraints={config_summary['num_constraints']}"
    )

    start = time.time()
    try:
        result = solve_trajectory_optimization(
            pipeline,
            code,
            task_name,
            args.iteration,
            output_dir,
        )
    except Exception as e:
        print(f"Optimization error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    elapsed = time.time() - start
    print(f"Optimization finished in {elapsed:.1f}s")

    if not args.skip_render:
        sim_result = execute_simulation(
            pipeline, result, args.iteration, output_dir
        )
        if sim_result.get("success"):
            print("Rendered planned trajectory video.")
        elif sim_result.get("debug_video_saved"):
            print(f"Rendered debug video: {sim_result.get('debug_video_path')}")
        elif sim_result.get("error"):
            print(f"Render: {sim_result['error']}")

    _print_optimization_summary(result, output_dir, args.iteration)
    return 0 if result.get("success") else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay MPC from saved constraint code (no LLM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--constraint-file",
        type=str,
        help="Path to saved configure_mpc() Python (e.g. results/example/sideflip.py)",
    )
    source.add_argument(
        "--constraint-from-summary",
        type=str,
        help="Extract code from summary_prompt_iter_*.txt CONSTRAINT CODE section",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mpc_replay",
        help="Directory for trajectories and videos (default: results/mpc_replay)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Iteration index used in output filenames (default: 1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["standard", "complementarity"],
        default=go2_config.CONSTRAINT_MODE,
        help="MPC physics constraint mode (default: %(default)s)",
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="Disable slack formulation (hard LLM constraints)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip planned/debug trajectory video rendering",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Print IPOPT iteration log (sets ipopt.print_level=5; default is quiet)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print Python tracebacks on errors (not IPOPT; use --solver-verbose for that)",
    )
    args = parser.parse_args()

    try:
        return run_replay(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
