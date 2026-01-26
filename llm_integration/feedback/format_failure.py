"""Format failure feedback for optimization failures."""

from typing import Any

import numpy as np


def generate_failure_feedback(
    iteration: int,
    command: str,
    optimization_metrics: dict[str, Any],
    constraint_violations: dict[str, Any],
    trajectory_analysis: dict[str, Any] | None,
    previous_constraints: str,
    state_traj: np.ndarray | None = None,
    initial_height: float = 0.2117,
) -> str:
    """
    Generate feedback when optimization fails to converge.

    This provides the LLM with actionable information about why the optimization
    failed and what to try differently.

    Args:
        iteration: Current iteration number
        command: The task command
        optimization_metrics: Solver metrics (iterations, error messages)
        constraint_violations: Detailed constraint violation info from MPC
        trajectory_analysis: Analysis of the (infeasible) trajectory if available
        previous_constraints: The constraint code that failed
        state_traj: The debug trajectory from failed optimization (if available)
        initial_height: Robot's initial COM height from config

    Returns:
        Formatted feedback string for the LLM
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} - OPTIMIZATION FAILED")
    lines.append("=" * 60)
    lines.append(f"Task: {command}")

    # Solver status
    lines.append("\n" + "-" * 60)
    lines.append("SOLVER STATUS")
    lines.append("-" * 60)
    lines.append("❌ OPTIMIZATION DID NOT CONVERGE")

    if optimization_metrics.get("solver_iterations"):
        lines.append(
            f"  Solver stopped after {optimization_metrics['solver_iterations']} iterations"
        )

    if optimization_metrics.get("error_message"):
        lines.append(f"  Error: {optimization_metrics['error_message']}")

    if optimization_metrics.get("infeasibility_info"):
        lines.append(f"  Infeasibility: {optimization_metrics['infeasibility_info']}")

    # Constraint violations analysis
    lines.append("\n" + "-" * 60)
    lines.append("CONSTRAINT VIOLATION ANALYSIS")
    lines.append("-" * 60)

    if constraint_violations.get("terminal_constraints"):
        lines.append("\n⚠️ TERMINAL CONSTRAINT VIOLATIONS:")
        for violation in constraint_violations["terminal_constraints"]:
            lines.append(f"  • {violation}")

    if constraint_violations.get("state_bounds"):
        lines.append("\n⚠️ STATE BOUND VIOLATIONS:")
        # Only show first 5 to avoid overwhelming
        for violation in constraint_violations["state_bounds"][:5]:
            lines.append(f"  • {violation}")
        if len(constraint_violations["state_bounds"]) > 5:
            lines.append(
                f"  ... and {len(constraint_violations['state_bounds']) - 5} more"
            )

    if constraint_violations.get("llm_constraints"):
        lines.append("\n🔴 YOUR LLM CONSTRAINT VIOLATIONS:")
        # Show first 10 LLM constraint violations
        for violation in constraint_violations["llm_constraints"][:10]:
            lines.append(f"  • {violation}")
        if len(constraint_violations["llm_constraints"]) > 10:
            lines.append(
                f"  ... and {len(constraint_violations['llm_constraints']) - 10} more"
            )

    if constraint_violations.get("llm_summary"):
        lines.append("\nLLM CONSTRAINT SUMMARY:")
        for summary in constraint_violations["llm_summary"]:
            lines.append(f"  • {summary}")

    if constraint_violations.get("summary"):
        lines.append("\nSYSTEM CONSTRAINT SUMMARY:")
        for summary in constraint_violations["summary"]:
            lines.append(f"  • {summary}")

    # Trajectory analysis from the failed attempt (if available)
    if trajectory_analysis and state_traj is not None and state_traj.size > 0:
        # Check if we have non-zero trajectory data
        if np.any(state_traj != 0):
            lines.append("\n" + "-" * 60)
            lines.append("FAILED TRAJECTORY ANALYSIS (solver's last attempt)")
            lines.append("-" * 60)
            lines.append(
                f"  Height range: {trajectory_analysis.get('min_com_height', 0):.3f}m - "
                f"{trajectory_analysis.get('max_com_height', 0):.3f}m"
            )
            lines.append(
                f"  Final height: {trajectory_analysis.get('final_com_height', 0):.3f}m"
            )
            lines.append(
                f"  Max pitch: {trajectory_analysis.get('max_pitch', 0):.3f}rad "
                f"({trajectory_analysis.get('max_pitch', 0) * 57.3:.1f}°)"
            )
            lines.append(
                f"  Max yaw: {trajectory_analysis.get('max_yaw', 0):.3f}rad "
                f"({trajectory_analysis.get('max_yaw', 0) * 57.3:.1f}°)"
            )

            # Analyze what the solver was trying to do
            if trajectory_analysis.get("max_com_height", 0) > 0.5:
                lines.append("\n  ℹ️ Solver was attempting a high jump")
            if abs(trajectory_analysis.get("max_yaw", 0)) > 0.5:
                lines.append("\n  ℹ️ Solver was attempting significant yaw rotation")

    # Common failure patterns and fixes
    lines.append("\n" + "-" * 60)
    lines.append("LIKELY CAUSES & SUGGESTED FIXES")
    lines.append("-" * 60)

    # Analyze the violations to give specific advice
    violations_text = str(constraint_violations)

    if "terminal" in violations_text.lower() or "landing" in violations_text.lower():
        lines.append("\n🔧 TERMINAL STATE CONFLICT:")
        lines.append("  Your constraints may conflict with landing requirements.")
        lines.append("  The robot MUST land with:")
        lines.append("    - vz in [-0.5, 0.3] m/s (not falling too fast)")
        lines.append("    - roll, pitch in [-0.2, 0.2] rad (upright)")
        lines.append("    - angular velocities in [-0.5, 0.5] rad/s")
        lines.append("  FIX: Relax your constraints near the end of the trajectory")
        lines.append("       or only apply constraints during flight phase.")

    if "underground" in violations_text.lower() or "height" in violations_text.lower():
        lines.append("\n🔧 HEIGHT CONSTRAINT CONFLICT:")
        lines.append("  Your constraints may be forcing impossible heights.")
        lines.append("  FIX: Ensure height constraints are achievable given")
        lines.append("       the contact sequence and physics.")

    if not constraint_violations.get(
        "terminal_constraints"
    ) and not constraint_violations.get("state_bounds"):
        lines.append("\n🔧 LLM CONSTRAINT LIKELY INFEASIBLE:")
        lines.append("  No obvious system constraint violations detected.")
        lines.append("  Your custom constraints are likely the issue.")
        lines.append("  Common problems:")
        lines.append(
            f"    1. Constraints at k=0 don't allow initial state (height={initial_height:.4f}m)"
        )
        lines.append(
            "    2. Mutually exclusive bounds (e.g., height>0.5 AND height<0.3)"
        )
        lines.append("    3. Constraints too tight - try loosening by 20-50%")
        lines.append("    4. Wrong timing - check contact_k for stance vs flight")

    # General advice
    lines.append("\n" + "-" * 60)
    lines.append("GENERAL DEBUGGING TIPS")
    lines.append("-" * 60)
    lines.append("1. START SIMPLE: Use only 1-2 constraints, not many")
    lines.append(
        f"2. CHECK k=0: Your constraints MUST allow height={initial_height:.4f}m at k=0"
    )
    lines.append("3. USE PHASES: Apply constraints only during relevant phases")
    lines.append("   Example: if contact_k.sum() == 0:  # Only during flight")
    lines.append("4. LOOSEN BOUNDS: If in doubt, make bounds 2x wider")
    lines.append("5. FINAL STATE ONLY: Consider constraining only the final state")
    lines.append("   Example: if k == horizon - 1:  # Only at the end")

    # Initial state reminder
    lines.append("\n" + "-" * 60)
    lines.append("REMINDER: INITIAL STATE")
    lines.append("-" * 60)
    lines.append(
        f"Robot starts at: height={initial_height:.4f}m, roll=0, pitch=0, yaw=0"
    )
    lines.append("Constraints at k=0 MUST allow this state!")

    # Previous code
    lines.append("\n" + "-" * 60)
    lines.append("PREVIOUS CODE (FAILED)")
    lines.append("-" * 60)
    lines.append(previous_constraints)

    # Instructions
    lines.append("\n" + "=" * 60)
    lines.append("TASK: Fix the constraints based on this failure analysis.")
    lines.append("Consider a COMPLETELY DIFFERENT approach if needed.")
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    return "\n".join(lines)
