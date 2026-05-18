"""Unified trajectory scoring via Claude LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import go2_config

from .format_hardness import format_hardness_report
from .format_metrics import format_trajectory_metrics_text
from .llm_evaluation import (
    DATA_DESCRIPTION,
    call_llm,
    extract_json_from_response,
    format_error_info,
    format_violations,
    save_prompt,
)


def _format_robot_context() -> str:
    """Format robot details into a context block for LLM prompts."""
    mass = go2_config.composite_mass
    height = float(go2_config.initial_crouch_qpos[2])
    upper = go2_config.urdf_joint_limits_upper.tolist()
    lower = go2_config.urdf_joint_limits_lower.tolist()
    grf_limit = float(go2_config.grf_limits)
    jvel_limit = float(max(go2_config.urdf_joint_velocities))
    mu = go2_config.experiment.mu_ground
    cl = go2_config.capability_limits

    return "\n".join([
        f"Unitree Go2 quadruped, mass {mass:.1f} kg, nominal standing height {height:.4f} m.",
        f"Per-component GRF limit: {grf_limit:.0f} N (fx,fy,fz each per foot).",
        f"Joint velocity limit: {jvel_limit:.1f} rad/s.",
        f"Ground friction coefficient: {mu}.",
        f"Joint limits: {lower[:3]} to {upper[:3]} (per leg, repeated x4).",
        "",
        "Physical capabilities (realistic for this robot):",
        f"- Max COM height gain: ~{cl['min_height_gain_normal']}-{cl['max_height_gain_normal']}m normal, ~{cl['max_height_gain_aggressive']}m aggressive",
        f"- Max realistic takeoff vz: ~{cl['min_takeoff_vz']}-{cl['max_takeoff_vz']} m/s",
        f"- Max realistic flight duration: ~{cl['min_flight_duration']}-{cl['max_flight_duration']}s",
        f"- Max realistic peak total GRF: ~{cl['min_peak_grf_total']:.0f}-{cl['max_peak_grf_total']:.0f} N ({cl['min_peak_grf_bodyweight_multiple']:.0f}-{cl['max_peak_grf_bodyweight_multiple']:.0f}x body weight)",
        "- Trajectories exceeding these limits are physically unrealizable even if the solver converges.",
    ])


def evaluate_iteration_unified(
    command: str,
    trajectory_analysis: dict[str, Any],
    constraint_code: str,
    opt_success: bool,
    error_info: dict[str, Any] | None = None,
    motion_quality_report: str = "",
    hardness_report: dict[str, Any] | None = None,
    mpc_dt: float | None = None,
    current_slack_weights: dict[str, float] | None = None,
    constraint_violations: dict[str, Any] | None = None,
    reference_analysis: str = "",
    run_dir: Path | None = None,
    iteration: int = 0,
) -> dict[str, Any]:
    """Score a trajectory. Returns dict with: score, criteria, warnings, summary."""
    if mpc_dt is None:
        raise ValueError("evaluate_iteration_unified: 'mpc_dt' must be provided.")

    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )
    robot_context = f"\n\n== ROBOT ==\n{_format_robot_context()}\n"

    _cl = go2_config.capability_limits
    _cap = (
        f"COM height gain > {_cl['max_height_gain_aggressive']}m, "
        f"takeoff vz > {_cl['max_takeoff_vz']} m/s, "
        f"total GRF > {_cl['max_peak_grf_total']:.0f} N, "
        f"or COM acceleration > {_cl['max_com_accel_typical_g']:.0f}g"
    )

    system_prompt = f"""You are an expert robotics engineer scoring a quadruped robot trajectory. Your job is ONLY to score — do not suggest fixes.{robot_context}

Read the task command to understand what was asked. Use ALL available data to determine what actually happened. Score how well the result matches the goal.

== DATA YOU RECEIVE ==

{DATA_DESCRIPTION}

== SCORING CRITERIA ==

TASK COMPLETION: Score proportional to how close the trajectory gets to the goal. Lower if motion quality is poor.

DIRECTION CHECK: The trajectory metrics include a "Direction summary" with signed rotation (roll/pitch/yaw) and translation (dx/dy/dz). Compare these signs against the task command's semantics. The reported magnitudes may look correct while the signs are opposite of what the task asks for — that is a failed task, not a success. Cap such cases at <=0.4 regardless of magnitude.

TERMINAL STATE CHECK: The trajectory must end in a stable deployable posture, not necessarily at the initial x/y position or initial yaw. Inspect final height, roll/pitch, linear velocity, angular velocity, joint feasibility, and contact/landing quality. Penalize unrecoverable endings: cap at <=0.6 if terminal linear/angular velocity is not small, roll/pitch is not upright, height is not near standing height, joints are far from a feasible stance, or feet are not planted for a ground-ending motion. Do NOT penalize terminal COM x/y displacement when the command asks for locomotion/translation; the robot should stay where it traveled. Do NOT penalize terminal yaw difference when the command asks for a heading change such as jump_180; it should hold the requested yaw instead of unwinding. For flips, require upright settled landing posture, but do not require the COM x/y or yaw to return to the exact start unless the command explicitly asks for that.

MOTION QUALITY: Use the motion quality report numbers to assess physical plausibility. Penalize physics exploits (phantom forces, energy from nowhere, feet below ground). Penalize exceeding capability limits: {_cap}.

SCORING GUIDE:
- 0.9-1.0: Task fully achieved, physically plausible, stable landing
- 0.7-0.9: Mostly achieved with minor quality issues
- 0.5-0.7: Partially achieved OR significant quality problems
- 0.3-0.5: Major shortfall or severely broken motion
- 0.0-0.3: Barely attempted or catastrophic failure

Score proportionally within bands. If solver FAILED, cap at 0.40 maximum.

Return JSON: {{"score": <float>, "criteria": [{{"name": "", "target": "", "achieved": "", "progress": 0.0}}], "warnings": [""], "summary": "<3-4 sentences>"}}
Return ONLY valid JSON."""

    metrics_text = format_trajectory_metrics_text(trajectory_analysis, opt_success)
    if opt_success:
        solver_status = "CONVERGED (success)"
    else:
        inf_pr = (error_info or {}).get("inf_pr")
        inf_du = (error_info or {}).get("inf_du")
        pr_str = f"{inf_pr:.2e}" if isinstance(inf_pr, (int, float)) else "n/a"
        du_str = f"{inf_du:.2e}" if isinstance(inf_du, (int, float)) else "n/a"
        solver_status = (
            f"FAILED (did not converge) — primal residual inf_pr={pr_str}, "
            f"dual residual inf_du={du_str}. Score is still capped at 0.40."
        )
    error_text = format_error_info(error_info)
    error_line = f"\n{error_text}" if error_text else ""

    user_message = f"""{"=" * 60}
                      TASK COMMAND
{"=" * 60}
{command}

{"=" * 60}
             SOLVER STATUS
{"=" * 60}
{solver_status}{error_line}

{"=" * 60}
          MOTION QUALITY REPORT
{"=" * 60}
{motion_quality_report or "Not available."}

{"=" * 60}
           TRAJECTORY METRICS
{"=" * 60}
{metrics_text}

{"=" * 60}
           CONSTRAINT HARDNESS
{"=" * 60}
{hardness_text or "Not available"}

{"=" * 60}
          CONSTRAINT VIOLATIONS
{"=" * 60}
{format_violations(constraint_violations)}

{"=" * 60}
           REFERENCE ANALYSIS
{"=" * 60}
{reference_analysis or "Not available"}

{"=" * 60}
            CONSTRAINT CODE
{"=" * 60}
{constraint_code}

Evaluate how well this trajectory achieves the commanded task."""

    save_prompt(run_dir, "scoring", iteration, system_prompt, user_message)

    response = call_llm(system_prompt, user_message)
    json_text = extract_json_from_response(response)
    result: dict[str, Any] = json.loads(json_text)
    if not opt_success and result.get("score", 0) > 0.4:
        result["score"] = 0.4
    return result
