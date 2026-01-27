"""Trajectory optimization functions for the feedback pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from utils import conversion
from utils.simulation import create_reference_trajectory

from ..logging_config import logger

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def solve_trajectory_optimization(
    self: "FeedbackPipeline",
    mpc_config_code: str,
    task_name: str,
    iteration: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Solve trajectory optimization with LLM-configured MPC."""

    if self.current_task_mpc is None:
        return {
            "success": False,
            "error": "No LLM MPC configured for this iteration",
            "converged": False,
        }

    # Get MPC configuration summary
    config_summary = self.current_task_mpc.get_configuration_summary()
    logger.info(
        f"MPC: {config_summary['task_name']}, {config_summary['duration']:.1f}s, {config_summary['num_constraints']} constraints"
    )

    # Setup initial conditions
    initial_state, _ = conversion.sim_to_mpc(
        self.config.experiment.initial_qpos, self.config.experiment.initial_qvel
    )

    ref = create_reference_trajectory(self.config.experiment.initial_qpos)

    try:
        state_traj, grf_traj, joint_vel_traj, status = (
            self.current_task_mpc.solve_trajectory(initial_state, ref)
        )

        if status == 0:
            logger.info("Optimization: converged")
        else:
            logger.warning(f"Optimization: failed (status {status})")

    except Exception as e:
        logger.error(f"Optimization error: {e}")

        # Fallback to default MPC
        try:
            state_traj, grf_traj, joint_vel_traj, status = (
                self.fallback_mpc.solve_trajectory(
                    initial_state, ref, self.config.mpc_config.contact_sequence
                )
            )
            logger.info(f"Fallback MPC status: {status}")
        except Exception as fallback_error:
            return {
                "success": False,
                "error": f"Both LLM MPC and fallback failed: {e}, {fallback_error}",
                "converged": False,
            }

    # Create metrics dict
    metrics = {
        "converged": status == 0,
        "status": "success" if status == 0 else "failed",
        "objective_value": 0.0,
        "mpc_type": "llm_configured"
        if self.current_task_mpc is not None
        else "fallback",
        "config_summary": config_summary,
        # Solver info for feedback
        "solver_iterations": getattr(self.current_task_mpc, "solver_iterations", None)
        if self.current_task_mpc
        else None,
        "error_message": getattr(self.current_task_mpc, "last_error", None)
        if self.current_task_mpc and status != 0
        else None,
        "infeasibility_info": getattr(self.current_task_mpc, "infeasibility_info", None)
        if self.current_task_mpc and status != 0
        else None,
    }

    # Analyze trajectory using LLM MPC time step
    mpc_dt = config_summary.get("time_step", self.config.mpc_config.mpc_dt)
    trajectory_analysis = self.constraint_generator.analyze_trajectory(
        state_traj, mpc_dt
    )

    # Log key trajectory metrics
    if status == 0:
        pitch = trajectory_analysis.get("total_pitch_rotation", 0)
        height = trajectory_analysis.get("height_gain", 0)
        logger.info(f"Trajectory: pitch={pitch:.2f}rad, height_gain={height:.2f}m")

    # Save trajectory data
    np.save(run_dir / f"state_traj_iter_{iteration}.npy", state_traj)
    np.save(run_dir / f"grf_traj_iter_{iteration}.npy", grf_traj)
    np.save(run_dir / f"joint_vel_traj_iter_{iteration}.npy", joint_vel_traj)

    result = {
        "success": status == 0,
        "status": status,
        "converged": status == 0,
        "state_trajectory": state_traj,
        "grf_trajectory": grf_traj,
        "joint_vel_trajectory": joint_vel_traj,
        "optimization_metrics": metrics,
        "trajectory_analysis": trajectory_analysis,
        "mpc_config_valid": True,
        "task_name": task_name,
        "mpc_config_code": mpc_config_code,
    }

    return result
