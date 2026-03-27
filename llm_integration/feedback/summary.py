"""Iteration summary generation via Claude LLM.

Produces a detailed performance summary that captures ALL metrics from the
motion quality report, trajectory metrics, hardness, violations, and reference
analysis in dense key=value format. This summary replaces sending raw reports
to the codegen LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    mpc_dt: float | None = None,
    current_slack_weights: dict[str, float] | None = None,
    reference_analysis: str = "",
    constraint_violations: dict[str, Any] | None = None,
    motion_quality_report: str = "",
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a detailed performance summary capturing ALL metrics.

    Returns a dict with structured fields. The codegen LLM reads these
    summaries instead of raw reports.
    """
    if mpc_dt is None:
        raise ValueError("generate_iteration_summary: 'mpc_dt' must be provided.")

    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )

    system_prompt = f"""You are summarizing a trajectory optimization iteration. A code-generation LLM will read this summary to understand what was tried and what happened. This summary must capture ALL information from the data. Be SPECIFIC with numbers. Use dense key=value format with | separators between sections.

== DATA YOU RECEIVE ==

{DATA_DESCRIPTION}
- ITERATION NUMBER, SCORE, SIMULATION RESULT.

Return a JSON object with these fields. Use dense key=value pairs, not prose:

{{
    "approach": "<2-3 lines: phase structure (names, durations, contacts), dt, horizon, constraint names and key bounds, reference trajectory targets, slack weights>",
    "solver": "<1 line: converged/failed, iteration count if failed, error type>",
    "motion_quality": "<1-3 dense lines, pipe-separated. ALL 10 sections: Smooth: jerk_rms=X max=X joint_max=X(jN,t=X) | Pen: foot=X body=X(link,depth,step) | GRF: phantom=X missing=X | Friction: violations=X worst_ratio=X | AngMom: flight_dev=X | Energy: disc=X% rate=X | Term: v=X w=X dh=X pitch=Xdeg | Contact: vz=X spread=X polygon=X | Joints: prox=X(jN,sN) | Manip: min=X pct_below=X>",
    "metrics": "<2-3 lines: height(init/max/gain/final), vel(max_com, terminal vx/vy/vz/wx/wy/wz), orient(roll/pitch/yaw max+total in deg), timing(dur, flight_start, flight_dur), GRF(max_total, max_foot, active%), actuator(max_jvel), phase(stance/flight steps, pitch per phase deg+%)>",
    "terminal": "<1 line: vx=X vy=X vz=X wx=X wy=X wz=X | height=X | roll=Xdeg pitch=Xdeg yaw=Xdeg>",
    "hardness": "<1-2 lines: per constraint — name max_slack=X total=X steps=X-X worst_t=X>",
    "violations": "<1-2 lines: terminal(x,y,z,vx,vy,vz,roll,pitch,yaw) | LLM constraints(name, count, max_dev, timesteps)>",
    "reference": "<1-2 lines: ref height/pitch/vz ranges, RMSE(height=X pitch=X vz=X), plausibility(max_vz_jump=X consistency=X)>"
}}

Do NOT include iteration, score, or success — set automatically.
Return ONLY valid JSON, no extra text."""

    metrics_text = format_trajectory_metrics_text(trajectory_analysis, opt_success)
    solver_status = "converged" if opt_success else "failed"
    error_text = format_error_info(error_info)
    error_line = f"\n{error_text}" if error_text else ""

    sim_text = ""
    if simulation_result:
        sim_success = simulation_result.get("success", False)
        sim_text = f"Rendering: {'success' if sim_success else 'failed'}"
        if simulation_result.get("error"):
            sim_text += f"\nRendering error: {simulation_result['error']}"

    user_message = f"""{"=" * 60}
                      TASK COMMAND
{"=" * 60}
{command}

Iteration: {iteration} | Score: {score:.2f} | {sim_text}

{"=" * 60}
             SOLVER STATUS
{"=" * 60}
{solver_status}{error_line}

{"=" * 60}
          MOTION QUALITY REPORT
{"=" * 60}
{motion_quality_report or "Not available"}

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
{constraint_code}"""

    save_prompt(run_dir, "summary", iteration, system_prompt, user_message)

    response = call_llm(system_prompt, user_message)
    json_text = extract_json_from_response(response)
    result: dict[str, Any] = json.loads(json_text)
    result["iteration"] = iteration
    result["score"] = score
    result["success"] = opt_success
    return result
