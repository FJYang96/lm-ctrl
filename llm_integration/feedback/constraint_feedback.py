"""Constraint-specific LLM feedback generation."""

from __future__ import annotations

from typing import Any

from ..logging_config import logger
from .llm_evaluation import get_evaluator


def generate_constraint_feedback(
    command: str,
    constraint_code: str,
    images: list[str] | None,
    visual_summary: str,
    hardness_report: str,
    constraint_violations: dict[str, Any],
    trajectory_analysis: dict[str, Any],
    opt_success: bool,
    error_info: dict[str, Any] | None,
    pivot_signal: str | None,
) -> str:
    """Generate constraint-specific feedback via LLM.

    Args:
        command: The task command
        constraint_code: Full constraint code from the LLM
        images: Video frames from the trajectory
        visual_summary: Text summary of the video frames
        hardness_report: Formatted constraint hardness analysis text
        constraint_violations: Dict of constraint violations
        trajectory_analysis: Trajectory metrics dict
        opt_success: Whether the solver converged
        error_info: Error information (if solver failed)
        pivot_signal: "pivot", "tweak", or None

    Returns:
        Multi-paragraph analysis text for the code-gen LLM
    """
    system_prompt = """You are an expert analyzing constraint design for quadruped MPC trajectory optimization.

Your job is to provide targeted feedback on the CONSTRAINT CODE specifically — what bounds are working,
what bounds are failing, and what changes to make.

=== CONSTRAINT-REFERENCE INTERPLAY ===

Constraints define the FEASIBLE REGION — the set of trajectories the solver is allowed to explore.
The reference trajectory provides the INITIAL GUESS — where the solver starts searching.

Key interactions:
- Tight constraints + bad reference = solver failure (starts outside feasible region, can't recover)
- Constraints with loopholes = no fix from reference (solver finds the easy way out regardless of starting point)
- Constraints that EXCLUDE the initial state at k=0 = immediate infeasibility
- Constraints must be CONTINUOUS across timesteps — no sudden jumps in bounds

Your feedback should focus ONLY on the constraints. Reference trajectory feedback is handled separately.

=== OUTPUT FORMAT ===

Write multi-paragraph analysis. Be specific about:
1. Which constraints are working (low slack, achieving intended effect)
2. Which constraints are failing (high slack, violated, or allowing loopholes)
3. Specific bound values to change and why
4. Timing issues (wrong phase, wrong timestep range)

Do NOT return JSON. Return readable analysis text."""

    mode_text = ""
    if pivot_signal == "pivot":
        mode_text = """MODE: MANDATORY PIVOT
The current constraint approach has stagnated or is declining. Suggest FUNDAMENTALLY DIFFERENT
constraint structures — different variables to constrain, different phase strategies, different
mathematical formulations. Do not suggest incremental adjustments."""
    elif pivot_signal == "tweak":
        mode_text = """MODE: ADJUSTMENT SUGGESTED
The current approach shows some promise. Suggest incremental changes — bound adjustments,
parameter tuning, timing shifts. Keep the overall constraint structure."""
    else:
        mode_text = """MODE: FIRST ITERATION
This is the first attempt. Analyze the constraint design and suggest improvements based
on the trajectory results."""

    # Format constraint violations
    violations_text = ""
    if constraint_violations:
        violation_lines = []
        for key, val in constraint_violations.items():
            if isinstance(val, list):
                for item in val:
                    violation_lines.append(f"  {key}: {item}")
            else:
                violation_lines.append(f"  {key}: {val}")
        violations_text = "\n".join(violation_lines) if violation_lines else "None"
    else:
        violations_text = "None"

    # Format trajectory metrics
    metrics_text = "No trajectory data available"
    if trajectory_analysis:
        ta = trajectory_analysis
        metrics_text = (
            f"Height: initial={ta.get('initial_com_height', 0):.3f}m, "
            f"max={ta.get('max_com_height', 0):.3f}m, gain={ta.get('height_gain', 0):.3f}m\n"
            f"Pitch: {ta.get('total_pitch_rotation', 0):.2f} rad "
            f"({abs(ta.get('total_pitch_rotation', 0)) * 57.3:.0f} deg)\n"
            f"Yaw: {ta.get('max_yaw', 0):.2f} rad, Roll: {ta.get('total_roll_rotation', 0):.2f} rad\n"
            f"Flight duration: {ta.get('flight_duration', 0):.2f}s\n"
            f"Max angular velocity: {ta.get('max_angular_vel', 0):.2f} rad/s\n"
            f"Final COM velocity: {ta.get('final_com_velocity', 0):.2f} m/s"
        )

    error_text = ""
    if error_info:
        err_parts = []
        if error_info.get("error_message"):
            err_parts.append(f"Error: {error_info['error_message']}")
        if error_info.get("solver_iterations"):
            err_parts.append(f"Solver iterations: {error_info['solver_iterations']}")
        error_text = "\n".join(err_parts)

    user_message = f"""COMMAND: {command}

{mode_text}

SOLVER STATUS: {"CONVERGED" if opt_success else "FAILED"}
{error_text}

CONSTRAINT CODE:
```python
{constraint_code}
```

TRAJECTORY METRICS:
{metrics_text}

CONSTRAINT VIOLATIONS:
{violations_text}

CONSTRAINT HARDNESS ANALYSIS:
{hardness_report if hardness_report else "Not available"}

VISUAL SUMMARY:
{visual_summary if visual_summary else "Not available"}

Provide targeted feedback on the constraint design."""

    try:
        evaluator = get_evaluator()
        response = evaluator._call_llm(system_prompt, user_message, images)
        return response.strip()
    except Exception as e:
        logger.error(f"Constraint feedback generation failed: {e}")
        return ""
