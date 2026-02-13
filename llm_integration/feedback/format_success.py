"""Format successful optimization feedback for the LLM."""

from typing import Any

from .format_metrics import (
    format_actuator_section,
    format_comparison_section,
    format_grf_section,
    format_phase_analysis_section,
    format_terminal_state_section,
    format_trajectory_metrics_section,
)
from .format_status import (
    format_footer_sections,
    format_mpc_config_section,
    format_optimization_status_section,
    format_simulation_status_section,
    format_task_progress_section,
)


def format_iteration_history(
    iteration_summaries: list[dict[str, Any]] | list[str] | None,
) -> list[str]:
    """Format detailed iteration history section."""
    lines: list[str] = []
    if not iteration_summaries or len(iteration_summaries) == 0:
        return lines

    lines.append("\n" + "-" * 60)
    lines.append("ITERATION HISTORY")
    lines.append("-" * 60)

    for item in iteration_summaries:
        # Handle both old format (strings) and new format (dicts)
        if isinstance(item, str):
            lines.append(f"  {item}")
        elif isinstance(item, dict):
            iter_num = item.get("iteration", "?")
            score = item.get("score", 0)
            success = item.get("success", False)
            status = "SUCCESS" if success else "FAILED"

            lines.append(f"\n  Iter {iter_num} [{status}] Score: {score:.2f}")

            # Show key metrics
            metrics = item.get("metrics", {})
            if metrics:
                pitch = metrics.get("pitch", 0)
                height = metrics.get("height_gain", 0)
                lines.append(
                    f"    Metrics: pitch={pitch:.2f}rad ({pitch * 57.3:.0f}°), height_gain={height:.3f}m"
                )

            # Show criteria evaluation
            criteria = item.get("criteria", [])
            if criteria:
                lines.append("    Criteria:")
                for c in criteria:
                    progress = c.get("progress", 0)
                    status_mark = "✓" if progress >= 0.8 else "✗"
                    lines.append(
                        f"      {status_mark} {c.get('name')}: {c.get('achieved')} ({progress:.0%})"
                    )

            # Show warnings
            warnings = item.get("warnings", [])
            if warnings:
                lines.append("    Warnings:")
                for w in warnings:
                    lines.append(f"      ⚠ {w}")

            # Show summary
            summary = item.get("summary", "")
            if summary:
                lines.append(f"    Summary: {summary}")

            # Show error for failed iterations
            error = item.get("error", "")
            if error and not success:
                lines.append(f"    Error: {error[:200]}")

    return lines


def format_enhanced_feedback(
    iteration: int,
    command: str,
    optimization_status: dict[str, Any],
    simulation_results: dict[str, Any],
    trajectory_analysis: dict[str, Any],
    phase_metrics: dict[str, Any],
    grf_metrics: dict[str, Any],
    actuator_metrics: dict[str, Any],
    task_progress: dict[str, Any],
    previous_constraints: str,
    previous_iteration_analysis: dict[str, Any] | None = None,
    initial_height: float = 0.2117,
    iteration_summaries: list[dict[str, Any]] | list[str] | None = None,
    pivot_signal: str | None = None,
) -> str:
    """
    Format all feedback into a structured string for the LLM.

    Args:
        initial_height: Robot's initial COM height from config
        iteration_summaries: List of LLM-generated summaries from previous iterations

    Returns:
        Formatted feedback string
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} FEEDBACK")
    lines.append("=" * 60)

    # Iteration History (what was tried before)
    lines.extend(format_iteration_history(iteration_summaries))

    # MPC Configuration Summary
    lines.extend(format_mpc_config_section(optimization_status))

    # Optimization status
    lines.extend(
        format_optimization_status_section(optimization_status, initial_height)
    )

    # Simulation status
    lines.extend(format_simulation_status_section(simulation_results))

    # Task Progress Table
    lines.extend(format_task_progress_section(task_progress))

    # Trajectory Metrics
    lines.extend(format_trajectory_metrics_section(trajectory_analysis))

    # Terminal State (for constraint tuning)
    lines.extend(format_terminal_state_section(trajectory_analysis))

    # Phase Analysis
    lines.extend(format_phase_analysis_section(phase_metrics))

    # GRF Analysis
    lines.extend(format_grf_section(grf_metrics))

    # Actuator Analysis
    lines.extend(format_actuator_section(actuator_metrics))

    # Comparison to previous iteration
    lines.extend(
        format_comparison_section(trajectory_analysis, previous_iteration_analysis)
    )

    # Pivot-aware strategy guidance (for converged-but-bad or stagnating)
    if pivot_signal == "pivot":
        lines.append("\n" + "!" * 60)
        lines.append("!" * 60)
        lines.append(
            "MANDATORY PIVOT: Despite convergence, the result is far from the goal."
        )
        lines.append(
            "You MUST start from scratch with a FUNDAMENTALLY DIFFERENT strategy:"
        )
        lines.append("")
        lines.append("  1. Use a COMPLETELY DIFFERENT contact sequence")
        lines.append("  2. Use DIFFERENT constraint types and bounds")
        lines.append(
            "  3. Generate a NEW reference trajectory with different parameters"
        )
        lines.append("  4. Do NOT make small tweaks — RETHINK the entire approach")
    elif pivot_signal == "tweak":
        lines.append("\n" + "-" * 60)
        lines.append(
            "ADJUSTMENT SUGGESTED: Score has not improved. Make targeted changes:"
        )
        lines.append("")
        lines.append("  1. Tighten the most important constraint(s) by 10-20%")
        lines.append("  2. Add one additional constraint to close loopholes")
        lines.append("  3. Keep the same general strategy but refine parameters")

    # Footer sections (initial state, previous code, instructions)
    lines.extend(format_footer_sections(initial_height, previous_constraints))

    return "\n".join(lines)
