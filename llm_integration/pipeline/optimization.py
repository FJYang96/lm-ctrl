"""Trajectory optimization functions for the feedback pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mpc.mpc_opti_slack import QuadrupedMPCOptiSlack
from utils import conversion
from utils.simulation import create_reference_trajectory

from ..logging_config import logger

if TYPE_CHECKING:
    from ..mpc.llm_task_mpc import LLMTaskMPC
    from .feedback_pipeline import FeedbackPipeline


def analyze_trajectory(
    state_traj: np.ndarray,
    mpc_dt: float,
    grf_traj: np.ndarray | None = None,
    joint_vel_traj: np.ndarray | None = None,
    contact_sequence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract key numeric metrics from a trajectory for LLM feedback.

    Args:
        state_traj: State trajectory array (N x 24)
        mpc_dt: Time step
        grf_traj: GRF trajectory (horizon x 12), 4 feet x 3 forces
        joint_vel_traj: Joint velocity trajectory (horizon x 12)
        contact_sequence: Contact sequence (4 x horizon), 1=contact, 0=flight per foot

    Returns:
        Dictionary of comprehensive trajectory metrics
    """
    if state_traj.shape[0] == 0:
        return {"error": "Empty trajectory"}

    try:
        # Extract key state components
        com_positions = state_traj[:, 0:3]  # x, y, z
        com_velocities = state_traj[:, 3:6]  # vx, vy, vz
        euler_angles = state_traj[:, 6:9]  # roll, pitch, yaw
        angular_velocities = state_traj[:, 9:12]  # wx, wy, wz
        joint_angles = state_traj[:, 12:24]  # 12 joint angles

        # Basic trajectory metrics
        metrics = {
            # Height analysis
            "max_com_height": float(np.max(com_positions[:, 2])),
            "min_com_height": float(np.min(com_positions[:, 2])),
            "initial_com_height": float(com_positions[0, 2]),
            "final_com_height": float(com_positions[-1, 2]),
            "height_gain": float(np.max(com_positions[:, 2]) - com_positions[0, 2]),
            # Orientation analysis
            "initial_roll": float(euler_angles[0, 0]),
            "final_roll": float(euler_angles[-1, 0]),
            "total_roll_rotation": float(euler_angles[-1, 0] - euler_angles[0, 0]),
            "initial_pitch": float(euler_angles[0, 1]),
            "final_pitch": float(euler_angles[-1, 1]),
            "total_pitch_rotation": float(euler_angles[-1, 1] - euler_angles[0, 1]),
            "initial_yaw": float(euler_angles[0, 2]),
            "final_yaw": float(euler_angles[-1, 2]),
            "total_yaw_rotation": float(euler_angles[-1, 2] - euler_angles[0, 2]),
            "max_roll": float(np.max(np.abs(euler_angles[:, 0]))),
            "max_pitch": float(np.max(np.abs(euler_angles[:, 1]))),
            "max_yaw": float(np.max(np.abs(euler_angles[:, 2] - euler_angles[0, 2]))),
            # Velocity analysis
            "max_com_velocity": float(np.max(np.linalg.norm(com_velocities, axis=1))),
            "max_angular_vel": float(
                np.max(np.linalg.norm(angular_velocities, axis=1))
            ),
            "final_com_velocity": float(np.linalg.norm(com_velocities[-1, :])),
            # Terminal velocity components (for terminal constraint tuning)
            "final_vx": float(com_velocities[-1, 0]),
            "final_vy": float(com_velocities[-1, 1]),
            "final_vz": float(com_velocities[-1, 2]),
            # Terminal angular velocity components
            "final_wx": float(angular_velocities[-1, 0]),
            "final_wy": float(angular_velocities[-1, 1]),
            "final_wz": float(angular_velocities[-1, 2]),
            # Displacement analysis
            "com_displacement_x": float(com_positions[-1, 0] - com_positions[0, 0]),
            "com_displacement_y": float(com_positions[-1, 1] - com_positions[0, 1]),
            "total_distance": float(
                np.sum(np.linalg.norm(np.diff(com_positions, axis=0), axis=1))
            ),
            # Timing
            "trajectory_duration": float(len(state_traj) * mpc_dt),
        }

        # Flight phase analysis
        initial_height = com_positions[0, 2]
        height_threshold = initial_height + 0.05  # 5cm above initial
        airborne_mask = com_positions[:, 2] > height_threshold

        if np.any(airborne_mask):
            flight_indices = np.where(airborne_mask)[0]
            metrics["flight_duration"] = float(len(flight_indices) * mpc_dt)
            metrics["flight_start_time"] = float(flight_indices[0] * mpc_dt)
            metrics["flight_peak_height"] = float(
                np.max(com_positions[flight_indices, 2])
            )
        else:
            metrics["flight_duration"] = 0.0
            metrics["flight_start_time"] = 0.0
            metrics["flight_peak_height"] = metrics["max_com_height"]

        # Joint motion analysis
        joint_ranges = np.max(joint_angles, axis=0) - np.min(joint_angles, axis=0)
        metrics["max_joint_range"] = float(np.max(joint_ranges))
        metrics["avg_joint_range"] = float(np.mean(joint_ranges))

        # Smoothness metrics
        com_accelerations = np.diff(com_velocities, axis=0) / mpc_dt
        if com_accelerations.shape[0] > 0:
            metrics["max_acceleration"] = float(
                np.max(np.linalg.norm(com_accelerations, axis=1))
            )
        else:
            metrics["max_acceleration"] = 0.0

        # GRF metrics (4 feet x 3 forces: fx, fy, fz per foot)
        if grf_traj is not None and grf_traj.shape[0] > 0:
            grf_z_indices = [2, 5, 8, 11]  # z-component per foot
            grf_z = grf_traj[:, grf_z_indices]  # (horizon, 4)
            total_grf_z = np.sum(grf_z, axis=1)  # (horizon,)
            metrics["max_total_grf_z"] = float(np.max(np.abs(total_grf_z)))
            metrics["mean_total_grf_z"] = float(np.mean(np.abs(total_grf_z)))
            metrics["max_single_foot_grf_z"] = float(np.max(np.abs(grf_z)))
            # Check GRF utilization: fraction of timesteps with significant GRF
            active_grf_mask = np.abs(total_grf_z) > 1.0  # > 1N threshold
            metrics["grf_active_fraction"] = float(np.mean(active_grf_mask))
        else:
            metrics["max_total_grf_z"] = 0.0
            metrics["mean_total_grf_z"] = 0.0
            metrics["max_single_foot_grf_z"] = 0.0
            metrics["grf_active_fraction"] = 0.0

        # Actuator (joint velocity) metrics
        if joint_vel_traj is not None and joint_vel_traj.shape[0] > 0:
            metrics["max_joint_velocity"] = float(np.max(np.abs(joint_vel_traj)))
            metrics["mean_joint_velocity"] = float(np.mean(np.abs(joint_vel_traj)))
            # Per-joint max velocity for utilization assessment
            per_joint_max = np.max(np.abs(joint_vel_traj), axis=0)
            metrics["joint_vel_utilization"] = float(np.mean(per_joint_max))
        else:
            metrics["max_joint_velocity"] = 0.0
            metrics["mean_joint_velocity"] = 0.0
            metrics["joint_vel_utilization"] = 0.0

        # Per-phase breakdown (stance vs flight)
        if (
            contact_sequence is not None
            and contact_sequence.shape[1] >= state_traj.shape[0] - 1
        ):
            horizon = state_traj.shape[0] - 1
            # A timestep is "flight" if all 4 feet are off ground
            all_flight = np.all(contact_sequence[:, :horizon] < 0.5, axis=0)
            # Map to state indices: state k+1 corresponds to timestep k
            flight_states = np.zeros(state_traj.shape[0], dtype=bool)
            flight_states[1:] = all_flight[:horizon]
            flight_states[0] = all_flight[0] if horizon > 0 else False
            stance_states = ~flight_states

            # Orientation changes per phase type
            for axis_name, axis_idx in [("roll", 0), ("pitch", 1), ("yaw", 2)]:
                angles = euler_angles[:, axis_idx]
                stance_change = 0.0
                flight_change = 0.0
                for k in range(1, len(angles)):
                    delta = abs(angles[k] - angles[k - 1])
                    if stance_states[k]:
                        stance_change += delta
                    else:
                        flight_change += delta
                metrics[f"{axis_name}_change_stance"] = float(stance_change)
                metrics[f"{axis_name}_change_flight"] = float(flight_change)

            # Max angular velocity per phase
            ang_vel_mag = np.linalg.norm(angular_velocities, axis=1)
            if np.any(stance_states):
                metrics["max_angular_vel_stance"] = float(
                    np.max(ang_vel_mag[stance_states])
                )
            else:
                metrics["max_angular_vel_stance"] = 0.0
            if np.any(flight_states):
                metrics["max_angular_vel_flight"] = float(
                    np.max(ang_vel_mag[flight_states])
                )
            else:
                metrics["max_angular_vel_flight"] = 0.0

            # Per-axis max angular velocity during flight
            for axis_name, axis_idx in [("wx", 0), ("wy", 1), ("wz", 2)]:
                if np.any(flight_states):
                    metrics[f"max_{axis_name}_flight"] = float(
                        np.max(np.abs(angular_velocities[flight_states, axis_idx]))
                    )
                else:
                    metrics[f"max_{axis_name}_flight"] = 0.0

            # Height change per phase
            stance_height_change = 0.0
            flight_height_change = 0.0
            heights = com_positions[:, 2]
            for k in range(1, len(heights)):
                delta = heights[k] - heights[k - 1]
                if stance_states[k]:
                    stance_height_change += delta
                else:
                    flight_height_change += delta
            metrics["height_change_stance"] = float(stance_height_change)
            metrics["height_change_flight"] = float(flight_height_change)

            # Phase duration summary
            n_flight = int(np.sum(all_flight))
            n_stance = int(horizon - n_flight)
            metrics["n_stance_steps"] = n_stance
            metrics["n_flight_steps"] = n_flight

        return metrics

    except Exception as e:
        return {"error": f"Trajectory analysis failed: {str(e)}"}


def _extract_hardness_report(
    task_mpc: LLMTaskMPC | None,
    mpc_dt: float,
) -> dict[str, dict[str, Any]] | None:
    """Extract and log constraint hardness report from the slack MPC.

    Args:
        task_mpc: LLMTaskMPC instance (may be None).
        mpc_dt: Timestep duration for time-based logging.

    Returns:
        Hardness report dict, or None if unavailable.
    """
    if task_mpc is None:
        return None

    use_slack = task_mpc.use_slack

    # Log objective value decomposition if available
    mpc = task_mpc.mpc
    if mpc is not None:
        mpc_type = type(mpc).__name__
        logger.info(f"MPC type: {mpc_type}")

        if isinstance(mpc, QuadrupedMPCOptiSlack) and mpc._last_solution is not None:
            obj_value = float(mpc._last_solution.value(mpc.opti.f))
            logger.info(f"Optimization objective value: {obj_value:.4f}")
            if use_slack:
                slack_cost = float(mpc._last_solution.value(mpc.slack_penalty_cost))
                logger.info(f"Slack penalty cost: {slack_cost:.4f}")
                logger.info(f"Base cost (without slack): {obj_value - slack_cost:.4f}")
        else:
            obj_value = float(mpc.opti.value(mpc.opti.f))
            logger.info(f"Optimization objective value: {obj_value:.4f}")

    hardness_report = task_mpc.last_hardness_report

    if hardness_report:
        logger.info("Constraint hardness analysis:")
        for name, metrics_data in hardness_report.items():
            max_slack = metrics_data.get("max_slack_Linf", 0)
            active_timesteps = metrics_data.get("active_timesteps", [])
            slack_by_timestep = metrics_data.get("slack_by_timestep", {})

            if max_slack > 0.1:
                icon = "CRITICAL"
            elif max_slack > 0.01:
                icon = "HIGH"
            elif max_slack > 1e-6:
                icon = "LOW"
            else:
                icon = "OK"

            logger.info(f"  [{icon}] {name}: Max={max_slack:.4f}")

            # Detailed timestep info for problematic constraints
            if max_slack > 0.01 and active_timesteps:
                sorted_steps = sorted(active_timesteps)
                start_t = sorted_steps[0] * mpc_dt
                end_t = (sorted_steps[-1] + 1) * mpc_dt
                logger.info(
                    f"      Violated at: t={start_t:.2f}-{end_t:.2f}s "
                    f"({len(active_timesteps)} timesteps)"
                )

                # Show worst timesteps
                if slack_by_timestep:
                    worst = sorted(slack_by_timestep.items(), key=lambda x: -x[1])[:3]
                    worst_str = ", ".join(
                        f"k={k}(t={k * mpc_dt:.2f}s, slack={v:.3f})"
                        for k, v in worst
                        if v > 1e-6
                    )
                    if worst_str:
                        logger.info(f"      Worst at: {worst_str}")
    elif use_slack:
        logger.warning("No hardness report available - slack analysis may have failed")

    return hardness_report


def solve_trajectory_optimization(
    self: FeedbackPipeline,
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
        f"MPC: {config_summary['task_name']}, "
        f"{config_summary['duration']:.1f}s, "
        f"{config_summary['num_constraints']} constraints"
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
        return {
            "success": False,
            "error": f"LLM MPC failed: {e}",
            "converged": False,
        }

    # Log reference trajectory metrics
    ref_trajectory_data = self.current_task_mpc.ref_trajectory_data
    if ref_trajectory_data is not None:
        X_ref = ref_trajectory_data["X_ref"]
        logger.info(
            f"Reference trajectory: "
            f"height range [{X_ref[2, :].min():.3f}, {X_ref[2, :].max():.3f}]m, "
            f"pitch range [{X_ref[7, :].min():.2f}, {X_ref[7, :].max():.2f}]rad"
        )

    # Extract hardness report from slack formulation
    mpc_dt_hardness = self.current_task_mpc.mpc_dt
    logger.info(f"Slack formulation enabled: {self.current_task_mpc.use_slack}")
    hardness_report = _extract_hardness_report(self.current_task_mpc, mpc_dt_hardness)

    # Extract objective value from solver
    mpc = self.current_task_mpc.mpc
    assert mpc is not None
    if isinstance(mpc, QuadrupedMPCOptiSlack) and mpc._last_solution is not None:
        objective_value = float(mpc._last_solution.value(mpc.opti.f))
    elif status == 0:
        objective_value = float(mpc.opti.value(mpc.opti.f))
    else:
        objective_value = float("inf")

    # Create metrics dict
    metrics = {
        "converged": status == 0,
        "status": "success" if status == 0 else "failed",
        "objective_value": objective_value,
        "mpc_type": "llm_configured",
        "config_summary": config_summary,
        "solver_iterations": self.current_task_mpc.solver_iterations,
        "error_message": self.current_task_mpc.last_error if status != 0 else None,
        "infeasibility_info": self.current_task_mpc.infeasibility_info
        if status != 0
        else None,
        "hardness_report": hardness_report,
    }

    # Analyze trajectory using LLM MPC time step
    mpc_dt = config_summary["time_step"]
    contact_seq = self.current_task_mpc.contact_sequence
    trajectory_analysis = analyze_trajectory(
        state_traj,
        mpc_dt,
        grf_traj=grf_traj,
        joint_vel_traj=joint_vel_traj,
        contact_sequence=contact_seq,
    )

    # Log key trajectory metrics
    if status == 0:
        pitch = trajectory_analysis["total_pitch_rotation"]
        height = trajectory_analysis["height_gain"]
        logger.info(f"Trajectory: pitch={pitch:.2f}rad, height_gain={height:.2f}m")

    # Save trajectory data
    np.save(run_dir / f"state_traj_iter_{iteration}.npy", state_traj)
    np.save(run_dir / f"grf_traj_iter_{iteration}.npy", grf_traj)
    np.save(run_dir / f"joint_vel_traj_iter_{iteration}.npy", joint_vel_traj)

    # Save contact sequence
    if self.current_task_mpc.contact_sequence is not None:
        np.save(
            run_dir / f"contact_sequence_iter_{iteration}.npy",
            self.current_task_mpc.contact_sequence,
        )

    return {
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
        "ref_trajectory_data": ref_trajectory_data,
    }
