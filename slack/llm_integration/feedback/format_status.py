"""Status and progress section formatters."""

from typing import Any


def format_mpc_config_section(optimization_status: dict[str, Any]) -> list[str]:
    """Format MPC configuration section."""
    lines = []
    if "config_summary" in optimization_status:
        config = optimization_status["config_summary"]
        lines.append("\n" + "-" * 60)
        lines.append("MPC CONFIGURATION")
        lines.append("-" * 60)
        lines.append(f"Task: {config.get('task_name', 'unknown')}")
        lines.append(f"Duration: {config.get('duration', 0):.2f}s")
        lines.append(f"Time step: {config.get('time_step', 0.02):.3f}s")
        lines.append(f"Horizon: {config.get('horizon', 0)} steps")
        lines.append(f"Constraints: {config.get('num_constraints', 0)}")
        if "contact_phases" in config:
            lines.append("Contact phases:")
            for phase in config["contact_phases"]:
                pattern = phase.get("contact_pattern", phase.get("pattern", []))
                phase_end = phase["start_time"] + phase["duration"]
                lines.append(
                    f"  {phase['phase_type']}: {phase['start_time']:.2f}-{phase_end:.2f}s {pattern}"
                )
    return lines


def format_optimization_status_section(
    optimization_status: dict[str, Any], initial_height: float
) -> list[str]:
    """Format optimization status section."""
    lines = []
    if optimization_status.get("converged", False):
        lines.append("\n✅ OPTIMIZATION: SUCCESS")
        if "solver_iterations" in optimization_status:
            lines.append(
                f"  Solver converged in {optimization_status['solver_iterations']} iterations"
            )
    else:
        lines.append("\n❌ OPTIMIZATION: FAILED")
        if (
            "error_message" in optimization_status
            and optimization_status["error_message"]
        ):
            lines.append(f"  Error: {optimization_status['error_message']}")
        if "solver_iterations" in optimization_status:
            lines.append(
                f"  Solver stopped at {optimization_status['solver_iterations']} iterations"
            )
        if "infeasibility_info" in optimization_status:
            lines.append(
                f"  Infeasibility: {optimization_status['infeasibility_info']}"
            )
        lines.append("")
        lines.append("COMMON FAILURE CAUSES:")
        lines.append(
            f"  1. Constraints violate initial state (t=0) - robot starts at height={initial_height:.4f}m"
        )
        lines.append(
            "  2. Mutually exclusive constraints (e.g., height>0.5 AND height<0.3)"
        )
        lines.append("  3. Contact sequence doesn't match constraint timing")
        lines.append("  4. Bounds too tight - try loosening by 20%")
        lines.append("")
        lines.append(
            "Fix: SIMPLIFY and LOOSEN constraints. Start with just one key constraint."
        )
        lines.append("")
        lines.append("⚠️  DON'T BE AFRAID TO CHANGE YOUR APPROACH/CODE DRASTICALLY:")
        lines.append(
            "  - If optimization failed, the constraint STRUCTURE may be fundamentally flawed"
        )
        lines.append(
            "  - Small tweaks to bad structure won't help - RETHINK the ENTIRE approach"
        )
        lines.append(
            "  - Consider: constrain ONLY the final state (e.g., final yaw = -π)"
        )
        lines.append(
            "  - Avoid progressive bounds that create 'traps' (loose early, tight late)"
        )
    return lines


def format_simulation_status_section(simulation_results: dict[str, Any]) -> list[str]:
    """Format simulation status section."""
    lines = []
    if simulation_results.get("success", False):
        tracking_error = simulation_results.get("tracking_error", 0)
        lines.append(f"✅ SIMULATION: SUCCESS (tracking error: {tracking_error:.3f})")
    else:
        lines.append("❌ SIMULATION: FAILED")
        if "error" in simulation_results:
            lines.append(f"  Error: {simulation_results['error']}")
        sim_analysis = simulation_results.get("simulation_analysis", {})
        if sim_analysis.get("issues"):
            lines.append("  Issues detected:")
            for issue in sim_analysis["issues"]:
                lines.append(f"    - {issue}")
    return lines


def format_task_progress_section(task_progress: dict[str, Any]) -> list[str]:
    """Format task progress table section."""
    lines = []
    lines.append("\n" + "-" * 60)
    lines.append("TASK PROGRESS")
    lines.append("-" * 60)
    lines.append(
        f"{'Criterion':<25} {'Required':<15} {'Achieved':<15} {'Progress':<10}"
    )
    lines.append("-" * 60)
    for criterion in task_progress.get("criteria", []):
        progress_pct = criterion["progress"] * 100
        status = "✓" if progress_pct >= 90 else "⚠️" if progress_pct >= 50 else "✗"
        lines.append(
            f"{criterion['name']:<25} {criterion['required']:<15} "
            f"{criterion['achieved']:<15} {progress_pct:>5.0f}% {status}"
        )
    lines.append("-" * 60)
    overall = task_progress.get("overall_progress", 0) * 100
    lines.append(f"{'OVERALL TASK COMPLETION:':<55} {overall:>5.0f}%")
    return lines


def format_footer_sections(
    initial_height: float, previous_constraints: str
) -> list[str]:
    """Format footer sections (initial state reminder, previous code, instructions)."""
    lines = []

    # Initial State Reminder
    lines.append("\n" + "-" * 60)
    lines.append("REMINDER: INITIAL STATE")
    lines.append("-" * 60)
    lines.append(
        f"Robot starts at: height={initial_height:.4f}m, roll=0, pitch=0, yaw=0"
    )
    lines.append("Constraints at k=0 MUST allow this state!")

    # Previous code
    lines.append("\n" + "-" * 60)
    lines.append("PREVIOUS CODE")
    lines.append("-" * 60)
    lines.append(previous_constraints)

    # Instructions
    lines.append("\n" + "=" * 60)
    lines.append("TASK: Generate improved constraints based on this feedback.")
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    return lines
