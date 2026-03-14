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
            status_label = "SUCCESS" if iter_success else "FAILED"
            lines.append(f"  Iter {iter_num} [{status_label}] Score: {iter_score:.2f}")

        # Recent iterations: full detail
        for entry in self.iteration_summaries[full_detail_start:]:
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

    lines.append("")
    lines.append("--- END OF ITERATION SUMMARIES ---")

    # === Current Iteration Detailed Results (XML diagnostic report) ===
    solver_status = "converged" if opt_success else "failed"
    lines.append("")
    lines.append("")
    lines.append(
        f'<diagnostic_report iteration="{iteration}" score="{score:.2f}" solver="{solver_status}">'
    )

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
            lines.append(f"  <error>{chr(10).join(error_parts)}</error>")

    # Full trajectory metrics
    if trajectory_analysis:
        metrics_lines = format_trajectory_metrics_section(
            trajectory_analysis, opt_success
        )
        lines.append("  <metrics>")
        for ml in metrics_lines:
            lines.append(f"  {ml}")
        lines.append("  </metrics>")

    # Full hardness report
    hardness_report = optimization_metrics.get("hardness_report")
    mpc_dt = float(self.config.mpc_config.mpc_dt)
    current_slack_weights = getattr(self, "current_slack_weights", None)
    hardness_text = format_hardness_report(
        hardness_report, dt=mpc_dt, current_slack_weights=current_slack_weights
    )
    if hardness_text:
        lines.append("  <hardness>")
        lines.append(f"  {hardness_text}")
        lines.append("  </hardness>")

    # Reference trajectory analysis (RMSE, plausibility)
    ref_trajectory_data = optimization_result.get("ref_trajectory_data")
    state_trajectory = optimization_result.get("state_trajectory")
    ref_analysis = _compute_reference_metrics(
        ref_trajectory_data, state_trajectory, mpc_dt
    )
    lines.append("  <reference_analysis>")
    lines.append(f"  {ref_analysis}")
    lines.append("  </reference_analysis>")

    # Constraint Code (ref stripped)
    constraint_only = strip_ref_trajectory_code(constraint_code)
    lines.append("  <constraint_code>")
    lines.append(f"  {constraint_only}")
    lines.append("  </constraint_code>")

    # Reference Trajectory Code
    ref_code = _extract_ref_trajectory_code(constraint_code)
    lines.append("  <reference_code>")
    if ref_code:
        lines.append(f"  {ref_code}")
    else:
        lines.append("  No reference trajectory function found in code.")
    lines.append("  </reference_code>")

    # Constraint Feedback
    lines.append("  <constraint_feedback>")
    if constraint_feedback:
        lines.append(f"  {constraint_feedback}")
    else:
        lines.append("  No constraint feedback available.")
    lines.append("  </constraint_feedback>")

    # Reference Trajectory Feedback
    lines.append("  <reference_feedback>")
    if reference_feedback:
        lines.append(f"  {reference_feedback}")
    else:
        lines.append("  No reference trajectory feedback available.")
    lines.append("  </reference_feedback>")

    # Visual Summary
    lines.append("  <visual_summary>")
    if visual_summary:
        lines.append(f"  {visual_summary}")
    else:
        lines.append("  No visual summary available.")
    lines.append("  </visual_summary>")

    lines.append("</diagnostic_report>")

    # === Footer ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("Generate improved constraints and reference trajectory.")
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    return "\n".join(lines)
