"""Codegen LLM call for constraint generation.

Single function ``generate_constraints`` → always returns ``(code, feedback_context)``.

* **Iteration 1**: ``feedback_context`` is ``None``.
* **Iteration 2+**: ``feedback_context`` is the assembled context string
  (built from the previous iteration's results).

Prompt templates live in ``client/prompts.py`` and are re-exported here.
"""

from __future__ import annotations

from typing import Any

from ..feedback.format_hardness import format_hardness_report
from ..feedback.format_metrics import format_trajectory_metrics_section
from ..feedback.llm_evaluation import call_llm
from ..feedback.reference_feedback import _compute_reference_metrics
from .code_extraction import extract_raw_code  # noqa: F401 (re-export)
from .prompts import (  # noqa: F401 (re-export)
    create_repair_prompt,
    get_system_prompt,
    get_user_prompt,
)


def generate_constraints(
    system_prompt: str,
    user_message: str,
    *,
    # --- feedback-context params (all optional, iteration 2+ only) ---
    iteration: int | None = None,
    command: str | None = None,
    optimization_result: dict[str, Any] | None = None,
    simulation_result: dict[str, Any] | None = None,
    constraint_code: str | None = None,
    run_dir: Any = None,
    iteration_summaries: list[dict[str, Any]] | None = None,
    mpc_dt: float | None = None,
    current_slack_weights: dict[str, float] | None = None,
    pivot_signal: str | None = None,
    feedback: str = "",
    score: float = 0.0,
    motion_quality_report: str = "",
) -> tuple[str, str | None]:
    """Generate optimization constraints using Claude.

    Args:
        system_prompt: System prompt for the LLM.
        user_message: User prompt (iteration 1) or repair prompt.
        iteration .. motion_quality_report: Feedback-context params from the
            previous iteration.  When ``iteration`` is not None the function
            builds the feedback context, prepends it, and calls the LLM.

    Returns:
        ``(llm_response, feedback_context)`` — ``feedback_context`` is ``None``
        on iteration 1.
    """
    # ---- Iteration 1: no feedback context ----
    if iteration is None:
        return call_llm(system_prompt, user_message), None

    # ---- Iteration 2+: build feedback context, then call LLM ----
    assert optimization_result is not None
    assert mpc_dt is not None
    assert iteration_summaries is not None
    assert constraint_code is not None
    opt_result = optimization_result
    opt_success = opt_result["success"]
    trajectory_analysis = opt_result["trajectory_analysis"]
    optimization_metrics = opt_result["optimization_metrics"]
    dt = mpc_dt

    lines: list[str] = []

    # === Header ===
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} FEEDBACK")
    lines.append("=" * 60)

    # === Terminology ===
    lines.append("")
    lines.append(
        "--- TERMINOLOGY (how labels and scores are defined below for iteration "
        "history and current iteration summary) ---"
    )
    lines.append(
        "SOLVER CONVERGED = the optimizer found a feasible solution that satisfies the "
        "constraints, but this does NOT mean the motion matches the task goal — a "
        "converged solver can still produce a trajectory that does nothing useful if "
        "the constraints have loopholes. "
        "SOLVER FAILED = the optimizer could not find any feasible solution within the "
        "constraint bounds. "
        "Score (0.0-1.0) is a separate LLM judgment of how well the actual trajectory "
        "matches the commanded task. "
        "Scores for failed solves are capped at 0.40 — prioritize getting the solver "
        "to converge first."
    )

    # === Iteration History (last 3 full, older one-line) ===
    summaries = iteration_summaries
    lines.append("")
    lines.append("--- ITERATION HISTORY ---")
    if summaries:
        total = len(summaries)
        lines.append(
            f"Total iterations so far: {total}. "
            f"Detailed analysis of iteration {iteration} follows below."
        )

        full_detail_start = max(0, total - 3)

        for i, entry in enumerate(summaries):
            if i >= full_detail_start:
                break
            status_label = "SOLVER CONVERGED" if entry["success"] else "SOLVER FAILED"
            lines.append(
                f"  Iter {entry['iteration']} [{status_label}] Score: {entry['score']:.2f}"
            )

        for entry in summaries[full_detail_start:]:
            status_label = "SOLVER CONVERGED" if entry["success"] else "SOLVER FAILED"
            lines.append("")
            lines.append(
                f"  Iter {entry['iteration']} [{status_label}] Score: {entry['score']:.2f}"
            )

            approach = entry["approach"]
            if approach:
                lines.append("    Approach:")
                for line in approach.split("\n"):
                    lines.append(f"      {line}")

            fb_summary = entry["feedback_summary"]
            if fb_summary:
                lines.append("    Feedback:")
                for line in fb_summary.split("\n"):
                    lines.append(f"      {line}")

            sim_summary = entry["simulation_summary"]
            if sim_summary:
                lines.append("    Simulation:")
                for line in sim_summary.split("\n"):
                    lines.append(f"      {line}")

            metrics_summary = entry["metrics_summary"]
            if metrics_summary:
                lines.append("    Metrics:")
                for line in metrics_summary.split("\n"):
                    lines.append(f"      {line}")
    else:
        lines.append("  No previous iterations.")

    lines.append("")
    lines.append("--- END OF ITERATION HISTORY ---")

    # === Mode ===
    lines.append("")
    if pivot_signal == "pivot":
        lines.append("--- MODE USED FOR THIS ITERATION: PIVOT ---")
        lines.append(
            "This iteration's code was generated under PIVOT mode — the approach had "
            "stagnated or declined, so a fundamentally different strategy was requested."
        )
    elif pivot_signal == "tweak":
        lines.append("--- MODE USED FOR THIS ITERATION: TWEAK ---")
        lines.append(
            "This iteration's code was generated under TWEAK mode — the approach showed "
            "progress, so incremental improvements were requested."
        )
    else:
        lines.append("--- MODE USED FOR THIS ITERATION: INITIAL ---")
        lines.append("This was the first iteration.")

    # === Current Iteration Detailed Analysis ===
    solver_label = "SOLVER CONVERGED" if opt_success else "SOLVER FAILED"
    lines.append("")
    lines.append("")
    lines.append("=" * 60)
    lines.append(
        f"  CURRENT ITERATION DETAILED ANALYSIS  [{solver_label}]  Score: {score:.2f}"
    )
    lines.append("=" * 60)

    if not opt_success:
        error_msg = optimization_metrics.get("error_message", "")
        solver_iters = optimization_metrics.get("solver_iterations")
        error_parts = []
        if error_msg:
            error_parts.append(error_msg)
        if solver_iters:
            error_parts.append(f"Solver iterations: {solver_iters}")
        if error_parts:
            lines.append("")
            lines.append("SOLVER FAILURE: " + " | ".join(error_parts))

    # === METRICS ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("              METRICS FOR THIS ITERATION")
    lines.append("=" * 60)

    if trajectory_analysis:
        metrics_lines = format_trajectory_metrics_section(
            trajectory_analysis, opt_success
        )
        for ml in metrics_lines:
            lines.append(ml)

    hardness_report = optimization_metrics["hardness_report"]
    hardness_text = format_hardness_report(
        hardness_report, dt=dt, current_slack_weights=current_slack_weights
    )
    if hardness_text:
        lines.append("")
        lines.append(hardness_text)

    ref_trajectory_data = opt_result["ref_trajectory_data"]
    state_trajectory = opt_result["state_trajectory"]
    ref_analysis = _compute_reference_metrics(ref_trajectory_data, state_trajectory, dt)
    lines.append("")
    lines.append(ref_analysis)

    # === MOTION QUALITY ANALYSIS ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("        MOTION QUALITY ANALYSIS FOR THIS ITERATION")
    lines.append("=" * 60)
    if motion_quality_report:
        lines.append(motion_quality_report)
    else:
        lines.append("No motion quality report available.")

    # === ENTIRE CODE ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("            ENTIRE CODE FOR THIS ITERATION")
    lines.append("=" * 60)
    lines.append(constraint_code)

    # === ENTIRE FEEDBACK ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("          ENTIRE FEEDBACK FOR THIS ITERATION")
    lines.append("=" * 60)
    if feedback:
        lines.append(feedback)
    else:
        lines.append(
            "No feedback available. Analyze the raw metrics, hardness data, "
            "and reference analysis above to diagnose issues yourself."
        )

    # === Footer ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("Generate improved constraints and reference trajectory.")
    lines.append(
        "Use ALL of the above — iteration history, previous summaries, current "
        "feedback, raw metrics, hardness data, violations, and reference analysis "
        "— as POWERFUL/IMPORTANT/GUIDING guidance. "
        "However, feel free to use your own independent decisions about what to "
        "change and why. "
        "If any of these are unavailable, use the rest to diagnose issues yourself."
    )
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    feedback_ctx = "\n".join(lines)
    full_message = f"{feedback_ctx}\n\n{user_message}"
    return call_llm(system_prompt, full_message), feedback_ctx
