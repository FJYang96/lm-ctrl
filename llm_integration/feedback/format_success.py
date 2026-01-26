"""Format successful optimization feedback for the LLM."""

from typing import Any

from .format_metrics import (
    format_actuator_section,
    format_comparison_section,
    format_grf_section,
    format_phase_analysis_section,
    format_trajectory_metrics_section,
)
from .format_status import (
    format_footer_sections,
    format_mpc_config_section,
    format_optimization_status_section,
    format_simulation_status_section,
    format_task_progress_section,
)


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
) -> str:
    """
    Format all feedback into a structured string for the LLM.

    Args:
        initial_height: Robot's initial COM height from config

    Returns:
        Formatted feedback string
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} FEEDBACK")
    lines.append("=" * 60)

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

    # Footer sections (initial state, previous code, instructions)
    lines.extend(format_footer_sections(initial_height, previous_constraints))

    return "\n".join(lines)
