"""Two Claude LLM calls per iteration (in pipeline execution order):

1. Scoring — scores the trajectory (receives motion quality report)
2. Summary — summarizes the iteration for history (receives simulation result)

Both share the same metrics data. Summary additionally receives
the simulation result from earlier pipeline steps.
Both share the same Claude client and helpers from llm_evaluation.py.
Motion quality is computed from trajectory data BEFORE these calls — the report is passed in.

The codegen LLM performs its own diagnosis directly from raw metrics —
there is no separate feedback LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path
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

# Shared across both calls — does NOT include trajectory frames
_DATA_DESCRIPTION = """- TASK COMMAND: The user's task description specifying what the robot should do (e.g. "jump 0.3m high" or "backflip").
- MOTION QUALITY REPORT: Computed metrics analyzing the physical quality of the trajectory — smoothness (jerk), ground penetration (foot and full-body link penetration), GRF-contact consistency, friction cone compliance, angular momentum conservation, energy continuity, terminal stability, contact quality (impact velocity, landing foot placement, foot spread, support polygon, landing joint limits), and joint feasibility. Each section provides raw numerical metrics. Use these to assess physical plausibility in context of the task.
- METRICS: Numerical trajectory data (positions, velocities, orientations, timing, GRF, actuator loads).
- HARDNESS DATA: Raw constraint slack values and violation timesteps. Larger slack = solver had to relax the constraint more.
- VIOLATIONS: Which constraints were violated and where.
- REFERENCE ANALYSIS: RMSE between reference and actual trajectory, plus plausibility metrics.
- CONSTRAINT CODE: The full constraint and reference trajectory code that produced this result.
- SOLVER STATUS: Whether the optimizer converged or failed. Includes error details if failed."""


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------


def _save_prompt(
    run_dir: Path | None,
    label: str,
    iteration: int,
    system_prompt: str,
    user_message: str,
) -> None:
    """Save a prompt to disk for debugging."""
    if run_dir is None:
        return
    path = run_dir / f"{label}_prompt_iter_{iteration}.txt"
    with open(path, "w") as f:
        f.write("=== SYSTEM PROMPT ===\n")
        f.write(system_prompt)
        f.write("\n\n=== USER MESSAGE ===\n")
        f.write(user_message)


def _format_robot_context(robot_details: dict[str, Any]) -> str:
    """Format robot details into a context block for LLM prompts."""
    mass = robot_details.get("mass", 15.019)
    height = robot_details.get("initial_height", 0.2117)
    lower = robot_details.get(
        "joint_limits_lower",
        [-0.8, -1.57, -2.6, -0.8, -1.57, -2.6, -0.8, -0.52, -2.6, -0.8, -0.52, -2.6],
    )
    upper = robot_details.get("joint_limits_upper", [0.8, 1.6, -0.84] * 4)
    grf_limits = robot_details.get("grf_limits")
    jvel_limits = robot_details.get("joint_velocity_limits")
    mu = robot_details.get("mu_ground")

    lines = [
        f"Unitree Go2 quadruped, mass {mass:.1f} kg, nominal standing height {height:.4f} m.",
    ]
    if grf_limits is not None:
        if isinstance(grf_limits, (int, float)):
            grf_limit = grf_limits
        else:
            grf_limit = grf_limits[2] if len(grf_limits) > 2 else max(grf_limits)
        lines.append(
            f"Per-component GRF limit: {grf_limit:.0f} N (fx,fy,fz each per foot)."
        )
    if jvel_limits is not None:
        if isinstance(jvel_limits, (int, float)):
            jvel_limit = abs(jvel_limits)
        else:
            jvel_limit = max(abs(v) for v in jvel_limits) if jvel_limits else 0
        lines.append(f"Joint velocity limit: {jvel_limit:.1f} rad/s.")
    if mu is not None:
        lines.append(f"Ground friction coefficient: {mu}.")
    lines.append(f"Joint limits: {lower[:3]} to {upper[:3]} (per leg, repeated x4).")
    lines.append("")
    lines.append("Physical capabilities (realistic for this robot):")
    lines.append("- Max COM height gain: ~0.15-0.25m normal, ~0.3m aggressive")
    lines.append("- Max realistic takeoff vz: ~1.8-2.5 m/s")
    lines.append("- Max realistic flight duration: ~0.3-0.5s")
    lines.append("- Max realistic peak total GRF: ~900-1200 N (6-8x body weight)")
    lines.append(
        "- Trajectories exceeding these limits are physically unrealizable even if the solver converges."
    )
    return "\n".join(lines)


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
    run_dir: Path | None = None,
    iteration: int = 0,
    robot_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a trajectory. Returns dict with: score, criteria, warnings, summary."""

    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )

    robot_context = ""
    if robot_details:
        robot_context = f"\n\n== ROBOT ==\n{_format_robot_context(robot_details)}\n"

    system_prompt = f"""You are an expert robotics engineer scoring a quadruped robot trajectory. Your job is ONLY to score — do not suggest fixes.{robot_context}

Read the task command to understand what was asked. Use ALL available data — the motion quality report, metrics, hardness data, violations, reference analysis, constraint code, and solver status — to determine what actually happened. Do not rely on any single source; cross-reference the motion quality report with the numerical metrics to build a complete picture. Score how well the result matches the goal.

== DATA YOU RECEIVE ==

{_DATA_DESCRIPTION}

== SCORING CRITERIA ==

The score must reflect BOTH task completion AND motion quality. These are equally important — a trajectory that hits the target numbers but is physically implausible is NOT a good trajectory.

TASK COMPLETION (does the trajectory achieve what was commanded?):
- The score should be roughly proportional to how close the trajectory gets to the commanded goal.
- Lower if the motion quality is poor.
- Partial completion should be scored proportionally, not rounded up to a "good enough" baseline.

MOTION QUALITY (is the trajectory physically plausible?):
- The motion quality report provides raw metrics for each aspect of motion quality. Use these numbers to assess physical plausibility in the context of the task.
- Consider what is physically unavoidable for the commanded task (e.g., a jump will have high landing impact velocity, high jerk at contact transitions, and energy injection during pushoff).
- Penalize issues that indicate the optimizer is exploiting physics (phantom forces, energy appearing from nowhere during flight, feet below ground) or that the motion would fail in reality (robot tipping over, joints locked at limits for extended periods).
- Penalize trajectories that exceed physical capability limits: COM height gain > 0.3m, takeoff vz > 2.5 m/s, total GRF > 1200 N, or COM acceleration > 6g indicate the optimizer is exploiting the model's lack of torque-based GRF limits. These trajectories cannot be reproduced on real hardware.

SCORING GUIDE:
- 0.9-1.0: Task fully achieved with physically plausible, smooth motion and stable landing
- 0.7-0.9: Task mostly achieved with minor quality issues
- 0.5-0.7: Task partially achieved OR achieved but with significant quality problems
- 0.3-0.5: Major shortfall in task completion or severely broken motion
- 0.0-0.3: Task barely attempted or catastrophic failure

IMPORTANT: Score proportionally within each band. Examples:
- 90% of target rotation with clean physics → 0.82
- 98% of target rotation with clean physics → 0.88
- 90% of target with major physics violations → 0.65
- 98% of target with major physics violations → 0.72
Do NOT round to band boundaries. A 2% improvement matters.

If the solver FAILED, cap the score at 0.40 maximum — an unconverged trajectory is physically meaningless. For converged trajectories with motion quality issues, use your judgment: consider whether an issue is inherent to the task (e.g., high landing velocity for a jump, jerk at takeoff/landing transitions, energy injection during pushoff) vs genuinely broken (e.g., robot falls over, feet phase through ground, phantom forces). Penalize genuine physical implausibility heavily, but do not penalize unavoidable physics.

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

    error_text = format_error_info(error_info)
    error_line = f"\n{error_text}" if error_text else ""

    user_message = f"""{"=" * 60}
                      TASK COMMAND
{"=" * 60}
{command}

{"=" * 60}
             SOLVER STATUS FOR THIS ITERATION
{"=" * 60}
{solver_status}{error_line}

{"=" * 60}
          MOTION QUALITY REPORT FOR THIS ITERATION
{"=" * 60}
{motion_quality_report if motion_quality_report else "No motion quality report available."}

{"=" * 60}
           TRAJECTORY METRICS FOR THIS ITERATION
{"=" * 60}
{metrics_text}

{"=" * 60}
           CONSTRAINT HARDNESS FOR THIS ITERATION
{"=" * 60}
{hardness_text if hardness_text else "Not available"}

{"=" * 60}
          CONSTRAINT VIOLATIONS FOR THIS ITERATION
{"=" * 60}
{format_violations(constraint_violations)}

{"=" * 60}
           REFERENCE ANALYSIS FOR THIS ITERATION
{"=" * 60}
{reference_analysis if reference_analysis else "Not available"}

{"=" * 60}
            CONSTRAINT CODE FOR THIS ITERATION
{"=" * 60}
{constraint_code}

Evaluate how well this trajectory achieves the commanded task."""

    _save_prompt(run_dir, "scoring", iteration, system_prompt, user_message)

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
# 2. Iteration summary
# ---------------------------------------------------------------------------


def generate_iteration_summary(
    command: str,
    iteration: int,
    score: float,
    constraint_code: str,
    trajectory_analysis: dict[str, Any],
    opt_success: bool,
    error_info: dict[str, Any] | None = None,
    simulation_result: dict[str, Any] | None = None,
    hardness_report: dict[str, Any] | None = None,
    mpc_dt: float = 0.02,
    current_slack_weights: dict[str, float] | None = None,
    reference_analysis: str = "",
    constraint_violations: dict[str, Any] | None = None,
    motion_quality_report: str = "",
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a structured iteration summary for the history log.

    Returns a structured dict with multi-line fields (prose paragraphs).
    """
    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )

    system_prompt = f"""You are summarizing a trajectory optimization iteration as a compact table for a history log. A code-generation LLM will read these summaries to understand what was tried and what happened. Be SPECIFIC with numbers — vague summaries are useless.

== DATA YOU RECEIVE ==

{_DATA_DESCRIPTION}
- ITERATION NUMBER, SCORE, SIMULATION RESULT from the current iteration.

Return a JSON object with these string fields — one per input data source. Each field is one line (~250 chars max), EXCEPT "approach" which can be 2-3 lines (~400 chars). No paragraphs, no bullet lists, no markdown.

{{
    "approach": "<2-3 lines: from CONSTRAINT CODE — describe this iteration's strategy: phase structure, durations, horizon, constraint variables and bounds, reference trajectory design>",
    "solver": "<one line: from SOLVER STATUS — converged/failed, iteration count if failed, error type>",
    "physics": "<one line: from MOTION QUALITY — friction violations, jerk, ground penetration, angular momentum conservation>",
    "metrics": "<one line: from TRAJECTORY METRICS — goal progress (rotation/distance/height achieved vs target), key velocities>",
    "terminal": "<one line: from TRAJECTORY METRICS — terminal velocities (vz, wz), final orientations (roll, pitch), final height>",
    "hardness": "<one line: from CONSTRAINT HARDNESS — violated steps / total, total slack, worst constraint and timestep>",
    "reference": "<one line: from REFERENCE ANALYSIS — height/pitch/vz RMSE, plausibility issues>"
}}

Do NOT include iteration, score, or success fields — those are set automatically.

Return ONLY valid JSON, no extra text."""

    metrics_text = format_trajectory_metrics_text(trajectory_analysis, opt_success)

    sim_text = ""
    if simulation_result:
        sim_success = simulation_result.get("success", False)
        sim_text = f"Rendering: {'success' if sim_success else 'failed'}"
        if simulation_result.get("error"):
            sim_text += f"\nRendering error: {str(simulation_result['error'])}"

    solver_status = "converged" if opt_success else "failed"

    error_text = format_error_info(error_info)
    error_line = f"\n{error_text}" if error_text else ""

    user_message = f"""{"=" * 60}
                      TASK COMMAND
{"=" * 60}
{command}

Iteration: {iteration} | Score: {score:.2f} | {sim_text}

{"=" * 60}
             SOLVER STATUS FOR THIS ITERATION
{"=" * 60}
{solver_status}{error_line}

{"=" * 60}
          MOTION QUALITY REPORT FOR THIS ITERATION
{"=" * 60}
{motion_quality_report if motion_quality_report else "Not available"}

{"=" * 60}
           TRAJECTORY METRICS FOR THIS ITERATION
{"=" * 60}
{metrics_text}

{"=" * 60}
           CONSTRAINT HARDNESS FOR THIS ITERATION
{"=" * 60}
{hardness_text if hardness_text else "Not available"}

{"=" * 60}
          CONSTRAINT VIOLATIONS FOR THIS ITERATION
{"=" * 60}
{format_violations(constraint_violations)}

{"=" * 60}
           REFERENCE ANALYSIS FOR THIS ITERATION
{"=" * 60}
{reference_analysis if reference_analysis else "Not available"}

{"=" * 60}
            CONSTRAINT CODE FOR THIS ITERATION
{"=" * 60}
{constraint_code}"""

    _save_prompt(run_dir, "summary", iteration, system_prompt, user_message)

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
            "solver": "",
            "physics": "",
            "metrics": "",
            "terminal": "",
            "hardness": "",
            "reference": "",
        }
