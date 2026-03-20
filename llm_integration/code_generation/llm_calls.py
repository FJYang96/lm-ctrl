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
from ..feedback.llm_evaluation import call_llm, format_violations
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
    score: float = 0.0,
    motion_quality_report: str = "",
    constraint_violations: dict[str, Any] | None = None,
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

    # === Task Command ===
    lines.append("=" * 60)
    lines.append("                      TASK COMMAND")
    lines.append("=" * 60)
    lines.append(command if command else "No task command")
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} CONTEXT")
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

    # === Iteration History (windowed: best + last 3) ===
    summaries = iteration_summaries
    lines.append("")
    lines.append("=" * 60)
    lines.append("                    ITERATION HISTORY")
    lines.append("=" * 60)
    if summaries:
        total = len(summaries)
        lines.append(
            f"Total iterations so far: {total}. "
            f"Detailed analysis of iteration {iteration} follows below."
        )

        # Find best iteration
        best_idx = max(range(total), key=lambda i: summaries[i].get("score", 0))
        # Last 3 iterations
        recent_start = max(0, total - 3)
        shown_indices = set(range(recent_start, total))
        shown_indices.add(best_idx)

        # One-line summary of skipped iterations
        skipped = [i for i in range(total) if i not in shown_indices]
        if skipped:
            skipped_scores = ", ".join(
                f"{summaries[i].get('score', 0):.2f}" for i in skipped
            )
            skipped_iters = (
                f"{skipped[0] + 1}-{skipped[-1] + 1}"
                if len(skipped) > 1
                else str(skipped[0] + 1)
            )
            lines.append(
                f"  Iterations {skipped_iters} omitted (scores: {skipped_scores})"
            )

        _table_fields = [
            ("Approach", "approach"),
            ("Solver", "solver"),
            ("Physics", "physics"),
            ("Metrics", "metrics"),
            ("Terminal", "terminal"),
            ("Hardness", "hardness"),
            ("Reference", "reference"),
        ]

        for idx in sorted(shown_indices):
            entry = summaries[idx]
            status_label = "SOLVER CONVERGED" if entry["success"] else "SOLVER FAILED"
            best_tag = " [BEST]" if idx == best_idx else ""
            lines.append("")
            lines.append(
                f"  Iter {entry['iteration']} [{status_label}] "
                f"Score: {entry['score']:.2f}{best_tag}"
            )
            for label, key in _table_fields:
                val = entry.get(key, "")
                if val:
                    lines.append(f"    {label:12s} {val}")
    else:
        lines.append("  No previous iterations.")

    lines.append("")
    lines.append("--- END OF ITERATION HISTORY ---")

    # === Big separator between history and current iteration ===
    solver_label = "SOLVER CONVERGED" if opt_success else "SOLVER FAILED"
    lines.append("")
    lines.append("")
    lines.append("#" * 60)
    lines.append("#" * 60)
    lines.append(f"       CURRENT ITERATION  [{solver_label}]  Score: {score:.2f}")
    lines.append("#" * 60)
    lines.append("#" * 60)

    # === Solver Status ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("             SOLVER STATUS FOR THIS ITERATION")
    lines.append("=" * 60)
    lines.append(solver_label)
    if not opt_success:
        error_msg = optimization_metrics.get("error_message", "")
        solver_iters = optimization_metrics.get("solver_iterations")
        error_parts = []
        if error_msg:
            error_parts.append(error_msg)
        if solver_iters:
            error_parts.append(f"Solver iterations: {solver_iters}")
        if error_parts:
            lines.append(" | ".join(error_parts))

    # === Motion Quality Report ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("          MOTION QUALITY REPORT FOR THIS ITERATION")
    lines.append("=" * 60)
    if motion_quality_report:
        lines.append(motion_quality_report)
    else:
        lines.append("No motion quality report available.")

    # === Trajectory Metrics ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("           TRAJECTORY METRICS FOR THIS ITERATION")
    lines.append("=" * 60)
    if trajectory_analysis:
        metrics_lines = format_trajectory_metrics_section(
            trajectory_analysis, opt_success
        )
        for ml in metrics_lines:
            lines.append(ml)
    else:
        lines.append(
            "No trajectory data available (solver failed before trajectory analysis)."
        )

    # === Constraint Hardness ===
    hardness_report = optimization_metrics["hardness_report"]
    hardness_text = format_hardness_report(
        hardness_report, dt=dt, current_slack_weights=current_slack_weights
    )
    lines.append("")
    lines.append("=" * 60)
    lines.append("           CONSTRAINT HARDNESS FOR THIS ITERATION")
    lines.append("=" * 60)
    lines.append(hardness_text if hardness_text else "Not available")

    # === Constraint Violations ===
    violations_text = format_violations(constraint_violations)
    lines.append("")
    lines.append("=" * 60)
    lines.append("          CONSTRAINT VIOLATIONS FOR THIS ITERATION")
    lines.append("=" * 60)
    lines.append(violations_text)

    # === Reference Analysis ===
    ref_trajectory_data = opt_result["ref_trajectory_data"]
    state_trajectory = opt_result["state_trajectory"]
    ref_analysis = _compute_reference_metrics(ref_trajectory_data, state_trajectory, dt)
    lines.append("")
    lines.append("=" * 60)
    lines.append("           REFERENCE ANALYSIS FOR THIS ITERATION")
    lines.append("=" * 60)
    lines.append(ref_analysis)

    # === Constraint Code ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("            CONSTRAINT CODE FOR THIS ITERATION")
    lines.append("=" * 60)
    lines.append(constraint_code)

    # === Footer ===
    lines.append("")
    lines.append("=" * 60)
    lines.append("Generate improved constraints and reference trajectory.")
    lines.append(
        "You have the full diagnosis above — metrics, physics analysis, "
        "constraint hardness, violations, and iteration history. "
        "You decide the strategy: tweak parameters, restructure constraints, "
        "change the phase structure, redesign the contact sequence, or "
        "rewrite from scratch. "
        "Use the iteration history to avoid repeating failed approaches. "
        "If scores are stagnating or the solver has failed 3+ consecutive "
        "iterations, do NOT keep tweaking parameters within the same "
        "structure — try a structurally different approach (different number "
        "of phases, different total duration, different contact sequence, "
        "different interpolation method, different constraint variables, "
        "or rewrite from scratch). "
        "If NO iteration has ever converged, drastically simplify: use a "
        "single loose envelope constraint with wide bounds, reduce slack "
        "weights, and remove any per-timestep corridor or schedule "
        "constraints. Get the solver to converge first, then tighten. "
        "A converged solution with imperfect task completion is far more "
        "valuable than an unconverged one."
    )
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    feedback_ctx = "\n".join(lines)
    full_message = f"{feedback_ctx}\n\n{user_message}"
    return call_llm(system_prompt, full_message), feedback_ctx
