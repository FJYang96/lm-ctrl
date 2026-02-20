"""Feedback context generation for the feedback pipeline — unified dual-feedback format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..feedback.code_utils import strip_ref_trajectory_code
from ..feedback.format_hardness import format_hardness_report
from ..feedback.format_metrics import format_trajectory_metrics_section
from ..feedback.reference_feedback import _compute_reference_metrics
from .utils import _extract_ref_trajectory_code

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
    constraint_feedback: str = "",
    reference_feedback: str = "",
    visual_summary: str = "",
    score: float = 0.0,
) -> str:
    """Create unified feedback context for the next LLM iteration.

    Single path for both success and failure — no branching.
    """
    opt_success = optimization_result.get("success", False)
    trajectory_analysis = optimization_result.get("trajectory_analysis", {})
    optimization_metrics = optimization_result.get("optimization_metrics", {})

    lines: list[str] = []

    # === Header ===
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} FEEDBACK")
    lines.append("=" * 60)

    # === Mode ===
    lines.append("")
    if pivot_signal == "pivot":
        lines.append("--- MODE: PIVOT ---")
        lines.append(
            "Your approach has stagnated or declined. You MUST try a fundamentally"
        )
        lines.append(
            "different strategy — different constraint structures, different variables,"
        )
        lines.append("different phase strategies. Do NOT make incremental changes.")
    elif pivot_signal == "tweak":
        lines.append("--- MODE: TWEAK ---")
        lines.append("Your approach shows progress. Make incremental improvements —")
        lines.append(
            "adjust bounds, tune parameters, refine timing. Keep the overall structure."
        )
    else:
        lines.append("--- MODE: INITIAL ---")
        lines.append("This is the first iteration. Review the results and improve.")

    # === Iteration History ===
    lines.append("")
    lines.append("--- ITERATION HISTORY ---")
    if self.iteration_summaries:
        for entry in self.iteration_summaries:
            iter_num = entry.get("iteration", "?")
            iter_score = entry.get("score", 0.0)
            iter_success = entry.get("success", False)
            status_label = "SUCCESS" if iter_success else "FAILED"
            lines.append("")
            lines.append(f"  Iter {iter_num} [{status_label}] Score: {iter_score:.2f}")

            constraint_approach = entry.get("constraint_approach", "")
            if constraint_approach:
                lines.append("    Constraint approach:")
                for line in constraint_approach.split("\n"):
                    lines.append(f"      {line}")

            reference_approach = entry.get("reference_approach", "")
            if reference_approach:
                lines.append("    Reference approach:")
                for line in reference_approach.split("\n"):
                    lines.append(f"      {line}")

            cfb_summary = entry.get("constraint_feedback_summary", "")
            if cfb_summary:
                lines.append("    Constraint feedback:")
                for line in cfb_summary.split("\n"):
                    lines.append(f"      {line}")

            rfb_summary = entry.get("reference_feedback_summary", "")
            if rfb_summary:
                lines.append("    Reference feedback:")
                for line in rfb_summary.split("\n"):
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

    # === Current Iteration Detailed Results ===
    lines.append("")
    lines.append("--- CURRENT ITERATION DETAILED RESULTS ---")
    lines.append(f"  Score: {score:.2f}")
    lines.append(f"  Solver: {'converged' if opt_success else 'FAILED'}")

    # Error info for failed iterations
    if not opt_success:
        error_msg = optimization_metrics.get("error_message", "")
        if error_msg:
            lines.append(f"  Error: {error_msg}")
        solver_iters = optimization_metrics.get("solver_iterations")
        if solver_iters:
            lines.append(f"  Solver iterations: {solver_iters}")

    # Full trajectory metrics
    if trajectory_analysis:
        metrics_lines = format_trajectory_metrics_section(trajectory_analysis)
        lines.extend(metrics_lines)

    # Full hardness report
    hardness_report = optimization_metrics.get("hardness_report")
    mpc_dt = float(self.config.mpc_config.mpc_dt)
    current_slack_weights = getattr(self, "current_slack_weights", None)
    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )
    if hardness_text:
        lines.append(hardness_text)

    # Reference trajectory analysis (RMSE, plausibility)
    ref_trajectory_data = optimization_result.get("ref_trajectory_data")
    state_trajectory = optimization_result.get("state_trajectory")
    ref_analysis = _compute_reference_metrics(
        ref_trajectory_data, state_trajectory, mpc_dt
    )
    lines.append("")
    lines.append("--- REFERENCE TRAJECTORY ANALYSIS ---")
    lines.append(ref_analysis)

    # === Constraint Code (ref stripped) ===
    lines.append("")
    lines.append("--- CONSTRAINT CODE (THIS ITERATION) ---")
    constraint_only = strip_ref_trajectory_code(constraint_code)
    lines.append(constraint_only)

    # === Reference Trajectory Code ===
    lines.append("")
    lines.append("--- REFERENCE TRAJECTORY CODE (THIS ITERATION) ---")
    ref_code = _extract_ref_trajectory_code(constraint_code)
    if ref_code:
        lines.append(ref_code)
    else:
        lines.append("  No reference trajectory function found in code.")

    # === Constraint Feedback ===
    lines.append("")
    lines.append("--- CONSTRAINT FEEDBACK ---")
    if constraint_feedback:
        lines.append(constraint_feedback)
    else:
        lines.append("  No constraint feedback available.")

    # === Reference Trajectory Feedback ===
    lines.append("")
    lines.append("--- REFERENCE TRAJECTORY FEEDBACK ---")
    if reference_feedback:
        lines.append(reference_feedback)
    else:
        lines.append("  No reference trajectory feedback available.")

    # === Visual Summary ===
    lines.append("")
    lines.append("--- VISUAL SUMMARY ---")
    if visual_summary:
        lines.append(visual_summary)
    else:
        lines.append("  No visual summary available.")

    # === Footer ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("Generate improved constraints and reference trajectory.")
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    return "\n".join(lines)
