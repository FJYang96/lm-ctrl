"""Unified feedback generation — single LLM call replacing separate constraint + reference calls."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..logging_config import logger
from .format_metrics import format_trajectory_metrics_text
from .llm_evaluation import get_evaluator
from .reference_feedback import _compute_reference_metrics


def generate_unified_feedback(
    command: str,
    constraint_code: str,
    visual_summary: str,
    hardness_report: str,
    constraint_violations: dict[str, Any],
    trajectory_analysis: dict[str, Any],
    opt_success: bool,
    error_info: dict[str, Any] | None,
    pivot_signal: str | None,
    ref_trajectory_data: dict[str, Any] | None,
    state_trajectory: np.ndarray | None,
    mpc_dt: float = 0.02,
) -> str:
    """Generate unified constraint + reference feedback via a single LLM call.

    Args:
        command: The task command
        constraint_code: Full constraint code from the LLM
        visual_summary: Text summary of the video frames
        hardness_report: Formatted constraint hardness analysis text
        constraint_violations: Dict of constraint violations
        trajectory_analysis: Trajectory metrics dict
        opt_success: Whether the solver converged
        error_info: Error information (if solver failed)
        pivot_signal: "pivot", "tweak", or None
        ref_trajectory_data: Dict with X_ref, U_ref arrays
        state_trajectory: Actual state trajectory (horizon+1, states_dim)
        mpc_dt: MPC time step in seconds

    Returns:
        Multi-paragraph prose analysis covering both constraints and reference trajectory
    """
    system_prompt = """You are an expert analyzing constraint and reference trajectory design for quadruped MPC trajectory optimization.

Your job is to provide a unified analysis of BOTH the constraint code and the reference trajectory, including how they interact.

CONSTRAINTS define the FEASIBLE REGION — the set of trajectories the solver is allowed to explore.
The REFERENCE TRAJECTORY provides the INITIAL GUESS — where the solver starts searching. It does NOT change the cost function.

Key interactions between constraints and reference:
- Tight constraints + bad reference = solver failure (starts outside feasible region, can't recover)
- Constraints with loopholes = no fix from reference (solver finds the easy way out regardless of starting point)
- Constraints that EXCLUDE the initial state at k=0 = immediate infeasibility
- Constraints must be CONTINUOUS across timesteps — no sudden jumps in bounds
- The reference should sit roughly in the CENTER of the constraint bounds
- If constraints force a specific rotation, the reference must show that rotation
- If constraints define a flight phase, the reference must have ballistic motion during that phase
- Phase timing in the reference must match the contact sequence exactly

Reference trajectory quality principles:
- Must be physically plausible (respect gravity, momentum conservation)
- Velocities must be consistent with positions (no teleportation)
- GRF should be zero during flight phases, ~mg/n_feet during stance
- Angular velocity during flight should be constant (momentum conservation)
- Angles should integrate from angular velocities

HARDNESS DATA: You will receive raw constraint hardness data (slack values, violation timesteps, worst offenders). You must assess severity from the raw numbers and determine what is critical vs acceptable. You must generate all recommendations.

PLAUSIBILITY DATA: You will receive raw plausibility metrics (velocity changes between timesteps, position-velocity consistency, RMSE comparisons). You must interpret these numbers to determine whether the reference trajectory is physically plausible.

OUTPUT FORMAT: Write detailed prose paragraphs. No markdown headers, no bullet lists, no asterisks, no code blocks. Just flowing analytical text organized as paragraphs. Cover constraints first, then reference trajectory, then their interaction. End with a final paragraph listing prioritized action items as numbered sentences (not bullets)."""

    mode_text = ""
    if pivot_signal == "pivot":
        mode_text = (
            "MODE: MANDATORY PIVOT\n"
            "The current approach has stagnated or is declining. Suggest FUNDAMENTALLY DIFFERENT\n"
            "constraint structures and reference trajectory shapes. Do not suggest incremental adjustments."
        )
    elif pivot_signal == "tweak":
        mode_text = (
            "MODE: ADJUSTMENT SUGGESTED\n"
            "The current approach shows some promise. Suggest incremental changes — bound adjustments,\n"
            "parameter tuning, timing shifts, reference peak tweaks. Keep the overall structure."
        )
    else:
        mode_text = (
            "MODE: FIRST ITERATION\n"
            "This is the first attempt. Analyze the constraint and reference design and suggest improvements\n"
            "based on the trajectory results."
        )

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
    metrics_text = format_trajectory_metrics_text(trajectory_analysis, opt_success)

    # Compute reference metrics
    ref_metrics = _compute_reference_metrics(
        ref_trajectory_data, state_trajectory, mpc_dt
    )

    error_text = ""
    if error_info:
        err_parts = []
        if error_info.get("error_message"):
            err_parts.append(f"Error: {error_info['error_message']}")
        if error_info.get("solver_iterations"):
            err_parts.append(f"Solver iterations: {error_info['solver_iterations']}")
        error_text = "\n".join(err_parts)

    solver_status = "converged" if opt_success else "failed"

    user_message = f"""<task>{command}</task>
<mode>{mode_text}</mode>
<solver status="{solver_status}">{error_text}</solver>
<constraint_code>
{constraint_code}
</constraint_code>
<metrics>
{metrics_text}
</metrics>
<violations>
{violations_text}
</violations>
<hardness>
{hardness_report if hardness_report else "Not available"}
</hardness>
<reference_analysis>
{ref_metrics}
</reference_analysis>
<visual_summary>
{visual_summary if visual_summary else "Not available"}
</visual_summary>

Provide unified feedback on both constraint design and reference trajectory."""

    try:
        evaluator = get_evaluator()
        response = evaluator._call_llm(system_prompt, user_message, None)
        return response.strip()
    except Exception as e:
        logger.error(f"Unified feedback generation failed: {e}")
        return ""
