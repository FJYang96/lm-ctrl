"""Feedback context generation for the feedback pipeline — unified format with 4 big sections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..feedback.format_hardness import format_hardness_report
from ..feedback.format_metrics import format_trajectory_metrics_section
from ..feedback.reference_feedback import _compute_reference_metrics

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def create_feedback_context(
    self: FeedbackPipeline,
    iteration: int,
    command: str,
    optimization_result: dict[str, Any],
    simulation_result: dict[str, Any],
    constraint_code: str,
    run_dir: Any,
    pivot_signal: str | None = None,
    feedback: str = "",
    score: float = 0.0,
    motion_quality_report: str = "",
) -> str:
    """Create unified feedback context for the next LLM iteration.

    Single path for both success and failure — no branching.
    Uses 4 big === sections: METRICS, MOTION QUALITY, ENTIRE CODE, ENTIRE FEEDBACK.
    """
    opt_success = optimization_result.get("success", False)
    trajectory_analysis = optimization_result.get("trajectory_analysis", {})
    optimization_metrics = optimization_result.get("optimization_metrics", {})

    lines: list[str] = []

    # === Header ===
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} FEEDBACK")
    lines.append("=" * 60)

    # === Terminology note (applies to both iteration history and detailed analysis) ===
    lines.append("")
    lines.append(
        "--- TERMINOLOGY (how labels and scores are defined below for iteration history and current iteration summary) ---"
    )
    lines.append(
        "SOLVER CONVERGED = the optimizer found a feasible solution that satisfies the constraints, "
        "but this does NOT mean the motion matches the task goal — a converged solver can still produce "
        "a trajectory that does nothing useful if the constraints have loopholes. "
        "SOLVER FAILED = the optimizer could not find any feasible solution within the constraint bounds. "
        "Score (0.0-1.0) is a separate LLM judgment of how well the actual trajectory matches the commanded task. "
        "Scores for failed solves are capped at 0.40 — prioritize getting the solver to converge first."
    )

    # === Iteration History (capped to last 3 full entries) ===
    lines.append("")
    lines.append("--- ITERATION HISTORY ---")
    if self.iteration_summaries:
        total = len(self.iteration_summaries)
        lines.append(
            f"Total iterations so far: {total}. "
            f"Detailed analysis of iteration {iteration} follows below."
        )

        # Determine cutoff: last 3 get full detail, older ones get one-line
        full_detail_start = max(0, total - 3)

        # Older iterations: one-line summary
        for i, entry in enumerate(self.iteration_summaries):
            if i >= full_detail_start:
                break
            iter_num = entry.get("iteration", "?")
            iter_score = entry.get("score", 0.0)
            iter_success = entry.get("success", False)
            status_label = "SOLVER CONVERGED" if iter_success else "SOLVER FAILED"
            lines.append(f"  Iter {iter_num} [{status_label}] Score: {iter_score:.2f}")

        # Recent iterations: full detail
        for entry in self.iteration_summaries[full_detail_start:]:
            iter_num = entry.get("iteration", "?")
            iter_score = entry.get("score", 0.0)
            iter_success = entry.get("success", False)
            status_label = "SOLVER CONVERGED" if iter_success else "SOLVER FAILED"
            lines.append("")
            lines.append(f"  Iter {iter_num} [{status_label}] Score: {iter_score:.2f}")

            approach = entry.get("approach", "")
            if approach:
                lines.append("    Approach:")
                for line in approach.split("\n"):
                    lines.append(f"      {line}")

            fb_summary = entry.get("feedback_summary", "")
            if fb_summary:
                lines.append("    Feedback:")
                for line in fb_summary.split("\n"):
                    lines.append(f"      {line}")

            sim_summary = entry.get("simulation_summary", "")
            if sim_summary:
                lines.append("    Simulation:")
                for line in sim_summary.split("\n"):
                    lines.append(f"      {line}")

            metrics_summary = entry.get("metrics_summary", "")
            if metrics_summary:
                lines.append("    Metrics:")
                for line in metrics_summary.split("\n"):
                    lines.append(f"      {line}")
    else:
        lines.append("  No previous iterations.")

    lines.append("")
    lines.append("--- END OF ITERATION HISTORY ---")

    # === Mode (used for this iteration's code generation) ===
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

    # Error info for failed iterations
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

    # ============================================================
    # METRICS
    # ============================================================
    lines.append("")
    lines.append("=" * 60)
    lines.append("              METRICS FOR THIS ITERATION")
    lines.append("=" * 60)

    # Trajectory metrics
    if trajectory_analysis:
        metrics_lines = format_trajectory_metrics_section(
            trajectory_analysis, opt_success
        )
        for ml in metrics_lines:
            lines.append(ml)

    # Hardness report
    hardness_report = optimization_metrics.get("hardness_report")
    mpc_dt = float(self.config.mpc_config.mpc_dt)
    current_slack_weights = getattr(self, "current_slack_weights", None)
    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )
    if hardness_text:
        lines.append("")
        lines.append(hardness_text)

    # Reference trajectory analysis (RMSE, plausibility)
    ref_trajectory_data = optimization_result.get("ref_trajectory_data")
    state_trajectory = optimization_result.get("state_trajectory")
    ref_analysis = _compute_reference_metrics(
        ref_trajectory_data, state_trajectory, mpc_dt
    )
    lines.append("")
    lines.append(ref_analysis)

    # ============================================================
    # MOTION QUALITY ANALYSIS
    # ============================================================
    lines.append("")
    lines.append("=" * 60)
    lines.append("        MOTION QUALITY ANALYSIS FOR THIS ITERATION")
    lines.append("=" * 60)
    if motion_quality_report:
        lines.append(motion_quality_report)
    else:
        lines.append("No motion quality report available.")

    # ============================================================
    # ENTIRE CODE
    # ============================================================
    lines.append("")
    lines.append("=" * 60)
    lines.append("            ENTIRE CODE FOR THIS ITERATION")
    lines.append("=" * 60)
    lines.append(constraint_code)

    # ============================================================
    # ENTIRE FEEDBACK
    # ============================================================
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
        "Use ALL of the above — iteration history, previous summaries, current feedback, "
        "raw metrics, hardness data, violations, and reference analysis — as POWERFUL/IMPORTANT/GUIDING guidance. "
        "However, feel free to use your own independent decisions about what to change and why. "
        "If any of these are unavailable, use the rest to diagnose issues yourself."
    )
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    return "\n".join(lines)
