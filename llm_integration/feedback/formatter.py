"""Feedback formatting utilities for enhanced LLM feedback.

This module provides the main entry point for feedback generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .format_failure import generate_failure_feedback
from .format_hardness import (
    format_hardness_report,
    format_hardness_suggestions,
    get_hardness_summary,
)
from .format_success import format_enhanced_feedback
from .task_progress import compute_task_progress
from .trajectory_analysis import (
    analyze_actuator_saturation,
    analyze_grf_profile,
    analyze_phase_metrics,
)

# Re-export for backwards compatibility
__all__ = [
    "format_enhanced_feedback",
    "generate_failure_feedback",
    "generate_enhanced_feedback",
    "format_hardness_report",
    "format_hardness_suggestions",
    "get_hardness_summary",
]


def generate_enhanced_feedback(
    iteration: int,
    command: str,
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    joint_torques_traj: np.ndarray | None,
    contact_sequence: np.ndarray,
    mpc_dt: float,
    optimization_status: dict[str, Any],
    simulation_results: dict[str, Any],
    trajectory_analysis: dict[str, Any],
    previous_constraints: str,
    previous_iteration_analysis: dict[str, Any] | None = None,
    robot_mass: float = 15.0,
    initial_height: float = 0.2117,
    iteration_summaries: list[dict[str, Any]] | list[str] | None = None,
    hardness_report: dict[str, dict[str, Any]] | None = None,
    current_slack_weights: dict[str, float] | None = None,
    pivot_signal: str | None = None,
) -> str:
    """
    Generate comprehensive enhanced feedback for the LLM.

    This is the main entry point that combines all analysis.

    Args:
        initial_height: Robot's initial COM height from config
        iteration_summaries: List of LLM-generated summaries from previous iterations
        hardness_report: Constraint hardness analysis from slack formulation
        current_slack_weights: Current slack weights set by LLM

    Returns:
        Formatted feedback string
    """
    # Phase analysis
    phase_metrics = analyze_phase_metrics(
        state_traj, contact_sequence, mpc_dt, grf_traj
    )

    # GRF analysis
    grf_metrics = analyze_grf_profile(grf_traj, contact_sequence, robot_mass)

    # Actuator analysis
    actuator_metrics = analyze_actuator_saturation(joint_vel_traj, joint_torques_traj)

    # Task progress
    task_progress = compute_task_progress(command, trajectory_analysis)

    # Format everything
    feedback = format_enhanced_feedback(
        iteration=iteration,
        command=command,
        optimization_status=optimization_status,
        simulation_results=simulation_results,
        trajectory_analysis=trajectory_analysis,
        phase_metrics=phase_metrics,
        grf_metrics=grf_metrics,
        actuator_metrics=actuator_metrics,
        task_progress=task_progress,
        previous_constraints=previous_constraints,
        previous_iteration_analysis=previous_iteration_analysis,
        initial_height=initial_height,
        iteration_summaries=iteration_summaries,
        pivot_signal=pivot_signal,
    )

    # Append constraint hardness analysis if available
    if hardness_report:
        hardness_section = format_hardness_report(
            hardness_report,
            dt=mpc_dt,
            current_slack_weights=current_slack_weights,
        )
        if hardness_section:
            feedback += hardness_section

    return feedback
