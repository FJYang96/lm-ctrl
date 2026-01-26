"""Feedback context generation for the feedback pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..feedback import generate_enhanced_feedback, generate_failure_feedback

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def create_feedback_context(
    self: "FeedbackPipeline",
    iteration: int,
    command: str,
    optimization_result: dict[str, Any],
    simulation_result: dict[str, Any],
    constraint_code: str,
    run_dir: Path,
) -> str:
    """Create enhanced feedback context for the next LLM iteration.

    If optimization failed, generates failure-specific feedback instead of raising.
    """
    # Check if optimization failed - generate failure feedback instead
    optimization_converged = optimization_result.get("converged", False)
    optimization_success = optimization_result.get("success", False)

    if not optimization_converged or not optimization_success:
        # Get constraint violations from MPC for detailed failure feedback
        constraint_violations: dict[str, Any] = {}
        if self.current_task_mpc and self.current_task_mpc.mpc:
            try:
                # Get system constraint violations (terminal, height bounds)
                constraint_violations = (
                    self.current_task_mpc.mpc.get_constraint_violations()
                )

                # Also get LLM constraint violations
                try:
                    X_debug = self.current_task_mpc.mpc.opti.debug.value(
                        self.current_task_mpc.mpc.X
                    )
                    U_debug = self.current_task_mpc.mpc.opti.debug.value(
                        self.current_task_mpc.mpc.U
                    )
                    llm_violations = (
                        self.current_task_mpc.evaluate_constraint_violations(
                            X_debug, U_debug
                        )
                    )
                    # Merge LLM violations into constraint_violations
                    constraint_violations["llm_constraints"] = llm_violations.get(
                        "llm_constraints", []
                    )
                    constraint_violations["llm_summary"] = llm_violations.get(
                        "summary", []
                    )
                except Exception as llm_e:
                    constraint_violations["llm_constraints"] = [
                        f"Could not evaluate LLM constraints: {llm_e}"
                    ]

            except Exception as e:
                constraint_violations = {
                    "summary": [f"Could not analyze violations: {e}"]
                }

        # Get whatever trajectory data we have (may be from debug values)
        state_traj = optimization_result.get("state_trajectory")
        trajectory_analysis = optimization_result.get("trajectory_analysis", {})
        optimization_metrics = optimization_result.get("optimization_metrics", {})

        print("📝 Generating failure feedback for LLM...")
        # Get initial height from config
        initial_height = float(self.config.experiment.initial_qpos[2])
        return generate_failure_feedback(
            iteration=iteration,
            command=command,
            optimization_metrics=optimization_metrics,
            constraint_violations=constraint_violations,
            trajectory_analysis=trajectory_analysis,
            previous_constraints=constraint_code,
            state_traj=state_traj,
            initial_height=initial_height,
        )

    # Optimization succeeded - generate normal enhanced feedback
    trajectory_analysis = optimization_result.get("trajectory_analysis")
    if trajectory_analysis is None:
        raise ValueError(
            "Enhanced feedback requires trajectory_analysis but it was not provided"
        )

    optimization_status = optimization_result.get("optimization_metrics")
    if optimization_status is None:
        raise ValueError(
            "Enhanced feedback requires optimization_metrics but it was not provided"
        )

    # Get trajectory data for enhanced analysis - REQUIRED for success case
    state_traj = optimization_result.get("state_trajectory")
    if state_traj is None or (hasattr(state_traj, "size") and state_traj.size == 0):
        raise ValueError(
            "Enhanced feedback requires state_trajectory but it was empty or not provided"
        )

    grf_traj = optimization_result.get("grf_trajectory")
    if grf_traj is None or (hasattr(grf_traj, "size") and grf_traj.size == 0):
        raise ValueError(
            "Enhanced feedback requires grf_trajectory but it was empty or not provided"
        )

    joint_vel_traj = optimization_result.get("joint_vel_trajectory")
    if joint_vel_traj is None or (
        hasattr(joint_vel_traj, "size") and joint_vel_traj.size == 0
    ):
        raise ValueError(
            "Enhanced feedback requires joint_vel_trajectory but it was empty or not provided"
        )

    # Get contact sequence from current MPC - REQUIRED
    if self.current_task_mpc and self.current_task_mpc.contact_sequence is not None:
        contact_sequence = self.current_task_mpc.contact_sequence
        mpc_dt = self.current_task_mpc.mpc_dt
    else:
        raise ValueError(
            "Enhanced feedback requires contact_sequence from LLM MPC but it was not configured"
        )

    # Simulation results - REQUIRED for success case
    if simulation_result is None:
        raise ValueError(
            "Enhanced feedback requires simulation_result but it was not provided"
        )

    # Generate enhanced feedback
    initial_height = float(self.config.experiment.initial_qpos[2])
    return generate_enhanced_feedback(
        iteration=iteration,
        command=command,
        state_traj=state_traj,
        grf_traj=grf_traj,
        joint_vel_traj=joint_vel_traj,
        joint_torques_traj=self.current_joint_torques,
        contact_sequence=contact_sequence,
        mpc_dt=mpc_dt,
        optimization_status=optimization_status,
        simulation_results=simulation_result,
        trajectory_analysis=trajectory_analysis,
        previous_constraints=constraint_code,
        previous_iteration_analysis=self.previous_iteration_analysis,
        robot_mass=self.config.robot_data.mass,
        initial_height=initial_height,
    )
