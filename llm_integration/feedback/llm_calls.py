"""Three Claude LLM calls per iteration (in pipeline execution order):

1. Scoring — scores the trajectory (receives motion quality report)
2. Feedback — actionable feedback on constraints + reference (receives motion quality report)
3. Summary — summarizes the iteration for history (receives feedback text + simulation result)

All three share the same metrics data. Summary additionally receives
the feedback output and simulation result from earlier pipeline steps.
All share the same Claude client and helpers from llm_evaluation.py.
Motion quality is computed from trajectory data BEFORE these calls — the report is passed in.
"""

from __future__ import annotations

import json
from typing import Any

from ..logging_config import logger
from .format_hardness import format_hardness_report
from .format_metrics import format_trajectory_metrics_text
from .llm_evaluation import (
    call_llm,
    extract_json_from_response,
    format_error_info,
    format_violations,
)

# ---------------------------------------------------------------------------
# Shared prompt constants
# ---------------------------------------------------------------------------

# Shared across all three calls — does NOT include trajectory frames
_DATA_DESCRIPTION = """- TASK COMMAND: The user's task description specifying what the robot should do (e.g. "jump 0.3m high" or "backflip").
- METRICS: Numerical trajectory data (positions, velocities, orientations, timing, GRF, actuator loads).
- HARDNESS DATA: Raw constraint slack values and violation timesteps. Larger slack = solver had to relax the constraint more.
- VIOLATIONS: Which constraints were violated and where.
- REFERENCE ANALYSIS: RMSE between reference and actual trajectory, plus plausibility metrics.
- CONSTRAINT CODE: The full constraint and reference trajectory code that produced this result.
- SOLVER STATUS: Whether the optimizer converged or failed. Includes error details if failed."""

# Prepended by scoring and feedback (which receive the motion quality report)
_MOTION_QUALITY_LINE = """- MOTION QUALITY REPORT: Computed metrics analyzing the physical quality of the trajectory — \
smoothness (jerk), ground penetration, GRF-contact consistency, friction cone compliance, \
angular momentum conservation, energy continuity, terminal stability, contact quality, and \
joint feasibility. Each section is rated OK/WARNING/CRITICAL. This is your primary source \
for motion quality assessment — it is computed directly from the trajectory data, not from \
visual inspection."""


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------


def evaluate_iteration_unified(
    command: str,
    trajectory_analysis: dict[str, Any],
    constraint_code: str,
    opt_success: bool,
    error_info: dict[str, Any] | None = None,
    motion_quality_report: str = "",
    hardness_report: dict[str, Any] | None = None,
    mpc_dt: float = 0.02,
    current_slack_weights: dict[str, float] | None = None,
    constraint_violations: dict[str, Any] | None = None,
    reference_analysis: str = "",
) -> dict[str, Any]:
    """Score a trajectory. Returns dict with: score, criteria, warnings, summary."""

    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )

    system_prompt = f"""You are an expert robotics engineer scoring a quadruped robot trajectory. Your job is ONLY to score — do not suggest fixes.

Read the task command to understand what was asked. Use ALL available data — the motion quality report, metrics, hardness data, violations, reference analysis, constraint code, and solver status — to determine what actually happened. Do not rely on any single source; cross-reference the motion quality report with the numerical metrics to build a complete picture. Score how well the result matches the goal.

== DATA YOU RECEIVE ==

{_MOTION_QUALITY_LINE}
{_DATA_DESCRIPTION}

== SCORING CRITERIA ==

The score must reflect BOTH task completion AND motion quality. These are equally important — a trajectory that hits the target numbers but is physically implausible is NOT a good trajectory.

TASK COMPLETION (does the trajectory achieve what was commanded?):
- The score should be roughly proportional to how close the trajectory gets to the commanded goal.
- Lower if the motion quality is poor.
- Partial completion should be scored proportionally, not rounded up to a "good enough" baseline.

MOTION QUALITY (is the trajectory physically plausible?):
- Check the motion quality report for CRITICAL or WARNING sections. CRITICAL issues in ground penetration, friction cone, or terminal stability indicate severe physical implausibility.
- Smoothness: high jerk values indicate discontinuous, non-smooth motion.
- GRF-contact consistency: phantom forces or missing forces indicate the optimizer is exploiting physics.
- Joint quality: joints near limits or torque infeasibility indicate an unrealizable trajectory.
- Manipulability: near-singularity configurations mean the robot cannot control its feet.

SCORING GUIDE:
- 0.9-1.0: Task fully achieved with physically plausible, smooth motion and stable landing
- 0.7-0.9: Task mostly achieved with minor quality issues
- 0.5-0.7: Task partially achieved OR achieved but with significant quality problems
- 0.3-0.5: Major shortfall in task completion or severely broken motion
- 0.0-0.3: Task barely attempted or catastrophic failure

If the solver FAILED, cap the score at 0.40 maximum. If the solver CONVERGED but the motion quality report shows CRITICAL issues (ground penetration, friction violations, terminal instability), cap the score at 0.30 maximum regardless of how close the numbers are to the target.

Return a JSON object:
{{
    "score": <float 0.0-1.0>,
    "criteria": [
        {{
            "name": "<what is being measured>",
            "target": "<numerical target from the command>",
            "achieved": "<actual result from metrics>",
            "progress": <float 0.0-1.0>
        }}
    ],
    "warnings": ["<technical warning>"],
    "summary": "<3-4 sentences explaining the score>"
}}

Return ONLY valid JSON, no markdown, no extra text."""

    metrics_text = format_trajectory_metrics_text(trajectory_analysis, opt_success)

    solver_status = (
        "CONVERGED (success)" if opt_success else "FAILED (did not converge)"
    )

    user_message = f"""<task>{command}</task>
<motion_quality>
{motion_quality_report if motion_quality_report else "No motion quality report available."}
</motion_quality>
<solver status="{solver_status}">{format_error_info(error_info)}</solver>
<metrics>
{metrics_text}
</metrics>
<constraint_code>
{constraint_code}
</constraint_code>
<hardness>
{hardness_text if hardness_text else "Not available"}
</hardness>
<violations>
{format_violations(constraint_violations)}
</violations>
<reference_analysis>
{reference_analysis if reference_analysis else "Not available"}
</reference_analysis>

Evaluate how well this trajectory achieves the commanded task."""

    try:
        response = call_llm(system_prompt, user_message)
        json_text = extract_json_from_response(response)
        result: dict[str, Any] = json.loads(json_text)
        # Cap failed iteration scores at 0.4
        if not opt_success and result.get("score", 0) > 0.4:
            result["score"] = 0.4
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Unified evaluation failed: invalid JSON - {e}")
        return _default_evaluation()
    except Exception as e:
        logger.error(
            f"Unified LLM evaluation failed: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return _default_evaluation()


def _default_evaluation() -> dict[str, Any]:
    """Return default evaluation when LLM fails."""
    return {
        "score": 0.0,
        "criteria": [],
        "warnings": ["Could not parse LLM evaluation"],
        "summary": "Evaluation failed - using default score of 0.0 (worst possible score)",
    }


# ---------------------------------------------------------------------------
# 2. Unified feedback
# ---------------------------------------------------------------------------


def generate_unified_feedback(
    command: str,
    constraint_code: str,
    motion_quality_report: str,
    hardness_report: dict[str, Any] | None,
    constraint_violations: dict[str, Any],
    trajectory_analysis: dict[str, Any],
    opt_success: bool,
    error_info: dict[str, Any] | None,
    pivot_signal: str | None,
    mpc_dt: float = 0.02,
    current_slack_weights: dict[str, float] | None = None,
    reference_analysis: str = "",
) -> str:
    """Generate unified constraint + reference feedback via a single LLM call.

    Returns multi-paragraph prose analysis covering both constraints and reference.
    """
    system_prompt = f"""You are an expert providing actionable feedback on constraint and reference trajectory code for quadruped MPC trajectory optimization. A code-generation LLM will read your feedback to decide what to change. Every problem you identify must come with a concrete fix.

== HOW CONSTRAINTS AND REFERENCE INTERACT ==

Constraints define the FEASIBLE REGION — the set of trajectories the solver is allowed to explore. The reference trajectory provides the INITIAL GUESS — where the solver starts searching. It does NOT change the cost function. They must be designed together:

- Tight constraints + bad reference = solver failure (starts outside feasible region, can't recover)
- Constraints with loopholes = solver cheats regardless of reference (finds the easiest path, not the intended one)
- Constraints that EXCLUDE the initial state at k=0 = immediate infeasibility
- Constraints must be CONTINUOUS across timesteps — sudden jumps in bounds cause solver failures
- The reference should sit roughly in the CENTER of the constraint bounds
- If constraints force a specific rotation, the reference must show that rotation or the solver starts far from feasible
- If constraints define a flight phase, the reference must have ballistic motion during that phase
- Phase timing in the reference must match the contact sequence exactly

When suggesting changes, consider both sides: a constraint fix may require a matching reference update, and vice versa. Never suggest changing one without considering the impact on the other.

== REFERENCE TRAJECTORY PHYSICS ==

The reference must be physically plausible or the solver starts from an unrealistic point:
- Velocities must be consistent with positions (position change = velocity * dt)
- GRF must be zero during flight phases, ~mg/n_feet during stance
- Angular velocity during flight must be constant (momentum conservation, no external torques)
- Angles must integrate from angular velocities — don't set angles without matching omega

== DATA YOU RECEIVE ==

{_MOTION_QUALITY_LINE}
{_DATA_DESCRIPTION}
- MODE: Whether this is the first iteration, an incremental tweak, or a mandatory pivot (fundamentally different approach). Guides how aggressive your suggestions should be.

Use ALL available data — the motion quality report, metrics, hardness data, violations, reference analysis, constraint code, and solver status — to form your feedback. Do not rely on any single source; cross-reference the motion quality report with the numerical metrics to identify root causes accurately.

SOLVER FAILURE: If the solver status is "failed", this is the most critical issue. The motion quality report describes the debug trajectory — the solver's best attempt before giving up, NOT a valid solution. The metrics may look completely wrong or unrelated to the goal. Focus on WHY the solver failed and how to make the problem feasible: loosen bounds, fix phase timing, fix reference trajectory. A failed solver means the feasible region is too small or the initial guess is too far from it.

== OUTPUT REQUIREMENTS ==

Your output must be ACTIONABLE FEEDBACK, not a summary. For every problem you identify, state what specific code is causing it, what concrete change to make (exact bound values, timing, parameters), and why that change will fix the problem.

Write detailed prose paragraphs. No markdown headers, no bullet lists, no asterisks, no code blocks. Start by describing what the motion quality report shows vs what the task requires — highlight any CRITICAL or WARNING sections and what they imply about the trajectory's physical plausibility. Then give focused paragraphs on the highest-impact constraint and reference fixes, always explaining how they interact. End with a final paragraph of numbered priority actions (most impactful first), each stating the exact change to make."""

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

    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )

    metrics_text = format_trajectory_metrics_text(trajectory_analysis, opt_success)
    solver_status = "converged" if opt_success else "failed"

    user_message = f"""<task>{command}</task>
<motion_quality>
{motion_quality_report if motion_quality_report else "No motion quality report available."}
</motion_quality>
<mode>{mode_text}</mode>
<solver status="{solver_status}">{format_error_info(error_info)}</solver>
<constraint_code>
{constraint_code}
</constraint_code>
<metrics>
{metrics_text}
</metrics>
<violations>
{format_violations(constraint_violations)}
</violations>
<hardness>
{hardness_text if hardness_text else "Not available"}
</hardness>
<reference_analysis>
{reference_analysis if reference_analysis else "Not available"}
</reference_analysis>

Provide unified feedback on both constraint design and reference trajectory. Start by describing what the motion quality report shows."""

    try:
        response = call_llm(system_prompt, user_message)
        return response.strip()
    except Exception as e:
        logger.error(f"Unified feedback generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# 3. Iteration summary
# ---------------------------------------------------------------------------


def generate_iteration_summary(
    command: str,
    iteration: int,
    score: float,
    constraint_code: str,
    feedback: str,
    trajectory_analysis: dict[str, Any],
    opt_success: bool,
    error_info: dict[str, Any] | None = None,
    simulation_result: dict[str, Any] | None = None,
    hardness_report: dict[str, Any] | None = None,
    mpc_dt: float = 0.02,
    current_slack_weights: dict[str, float] | None = None,
    reference_analysis: str = "",
    constraint_violations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a structured iteration summary for the history log.

    Returns a structured dict with multi-line fields (prose paragraphs).
    """
    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )

    system_prompt = f"""You are summarizing a trajectory optimization iteration for a history log. A code-generation LLM will read this summary to understand what was tried, what happened, and what to do differently next time. Be DETAILED and SPECIFIC — vague summaries are useless.

== DATA YOU RECEIVE ==

{_DATA_DESCRIPTION}
- ITERATION NUMBER: Which iteration this is (0-indexed).
- SCORE: The LLM-assigned score (0.0-1.0) for how well the trajectory matches the task goal. This is set automatically — do not include it in your output.
- SIMULATION RESULT: Whether MuJoCo rendering succeeded or failed, and what the robot actually did.
- FEEDBACK: The full unified feedback output from this iteration — actionable analysis covering constraint design, reference trajectory issues, and prioritized fixes. Summarize its key points, do not repeat it verbatim.

Use ALL available data — metrics, hardness data, violations, reference analysis, constraint code, solver status, simulation result, and feedback — to build a comprehensive summary. Cross-reference multiple sources to ensure accuracy.

Return a JSON object with ONLY these 4 string fields:
{{
    "approach": "<prose paragraph>",
    "feedback_summary": "<prose paragraph>",
    "simulation_summary": "<prose paragraph>",
    "metrics_summary": "<prose paragraph>"
}}

Do NOT include iteration, score, or success fields — those are set automatically.

All fields must be prose paragraphs. No markdown, no bullet lists, no asterisks, no code blocks.

approach: Describe the constraint and reference strategy. Include contact sequence phases and durations, constraint variables with bounds and timing, and reference trajectory design with numerical targets. Explain how constraints and reference work together.

feedback_summary: Summarize the unified feedback — what worked, what failed, root causes, and the prioritized fixes with concrete numbers.

simulation_summary: Whether rendering succeeded or failed, what the robot actually did, landing quality, any visible issues.

metrics_summary: Key numbers from trajectory metrics, hardness data, and reference analysis — rotation (rad and degrees), height, flight duration, solver status, terminal velocities, constraint slack values, reference RMSE.

Return ONLY valid JSON, no extra text."""

    metrics_text = format_trajectory_metrics_text(trajectory_analysis, opt_success)

    sim_text = ""
    if simulation_result:
        sim_success = simulation_result.get("success", False)
        sim_text = f"Rendering: {'success' if sim_success else 'failed'}"
        if simulation_result.get("error"):
            sim_text += f"\nRendering error: {str(simulation_result['error'])}"

    solver_status = "converged" if opt_success else "failed"

    user_message = f"""<task>{command}</task>
<iteration>{iteration}</iteration>
<score>{score:.2f}</score>
<solver status="{solver_status}">{format_error_info(error_info)}</solver>
<metrics>
{metrics_text}
</metrics>
<hardness>
{hardness_text if hardness_text else "Not available"}
</hardness>
<violations>
{format_violations(constraint_violations)}
</violations>
<reference_analysis>
{reference_analysis if reference_analysis else "Not available"}
</reference_analysis>
<simulation>{sim_text}</simulation>
<constraint_code>
{constraint_code}
</constraint_code>
<feedback>
{feedback if feedback else "None"}
</feedback>"""

    try:
        response = call_llm(system_prompt, user_message)
        json_text = extract_json_from_response(response)
        result: dict[str, Any] = json.loads(json_text)
        # Force ground-truth fields — LLM must not override these
        result["iteration"] = iteration
        result["score"] = score
        result["success"] = opt_success
        return result
    except Exception as e:
        logger.error(f"Iteration summary generation failed: {e}")
        return {
            "iteration": iteration,
            "score": score,
            "success": opt_success,
            "approach": "Summary generation failed",
            "feedback_summary": "",
            "simulation_summary": "",
            "metrics_summary": metrics_text,
        }
