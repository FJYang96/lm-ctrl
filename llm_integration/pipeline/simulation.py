"""Simulation execution functions for the feedback pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from utils import conversion
from utils.inv_dyn import compute_joint_torques
from utils.simulation import simulate_trajectory
from utils.visualization import render_and_save_planned_trajectory

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def execute_simulation(
    self: "FeedbackPipeline",
    optimization_result: dict[str, Any],
    iteration: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Execute the optimized trajectory in simulation."""

    if not optimization_result["success"]:
        return {
            "success": False,
            "error": "Cannot simulate - optimization failed",
            "tracking_error": float("inf"),
        }

    if self.env is None:
        return {
            "success": False,
            "error": "Cannot simulate - environment not available",
            "tracking_error": float("inf"),
        }

    try:
        state_traj = optimization_result["state_trajectory"]
        grf_traj = optimization_result["grf_trajectory"]
        joint_vel_traj = optimization_result["joint_vel_trajectory"]

        # Create input trajectory for rendering
        input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)

        print("🎬 Creating planned trajectory visualization...")
        # Render planned trajectory
        planned_traj_images = render_and_save_planned_trajectory(
            state_traj, input_traj, self.env, f"_iter_{iteration}"
        )

        print("🔧 Computing joint torques via inverse dynamics...")
        # Use LLM MPC's contact sequence if available
        if self.current_task_mpc and self.current_task_mpc.contact_sequence is not None:
            contact_seq = self.current_task_mpc.contact_sequence
            mpc_dt = self.current_task_mpc.mpc_dt
        else:
            contact_seq = self.config.mpc_config.contact_sequence
            mpc_dt = self.config.mpc_config.mpc_dt

        # Compute joint torques using inverse dynamics
        joint_torques_traj = compute_joint_torques(
            self.kindyn_model,
            state_traj,
            grf_traj,
            contact_seq,
            mpc_dt,
            joint_vel_traj,
        )

        # Store for enhanced feedback
        self.current_joint_torques = joint_torques_traj

        print("🏃 Executing trajectory in physics simulation...")
        # Execute in simulation
        qpos_traj, qvel_traj, sim_grf_traj, sim_images = simulate_trajectory(
            self.env, joint_torques_traj, planned_traj_images
        )

        # Calculate tracking error
        tracking_error = calculate_tracking_error(
            self, state_traj, qpos_traj, qvel_traj
        )

        # Analyze simulation results
        simulation_analysis = analyze_simulation_quality(
            self, qpos_traj, qvel_traj, sim_grf_traj, tracking_error
        )

        # Save simulation video
        print("💾 Saving simulation video...")
        if sim_images:
            import imageio

            fps = 1 / self.config.experiment.sim_dt
            video_path = run_dir / f"simulation_iter_{iteration}.mp4"
            imageio.mimsave(str(video_path), sim_images, fps=fps)
            print(f"  Saved: {video_path}")

        # Also save planned trajectory video if available
        if planned_traj_images:
            import imageio

            fps = 1 / self.config.experiment.sim_dt
            planned_video_path = run_dir / f"planned_traj_iter_{iteration}.mp4"
            imageio.mimsave(str(planned_video_path), planned_traj_images, fps=fps)
            print(f"  Saved: {planned_video_path}")

        result = {
            "success": True,
            "tracking_error": tracking_error,
            "simulation_analysis": simulation_analysis,
            "simulated_qpos": qpos_traj,
            "simulated_qvel": qvel_traj,
            "simulated_grf": sim_grf_traj,
            "realistic": simulation_analysis.get("realistic", False),
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Simulation failed: {str(e)}",
            "tracking_error": float("inf"),
        }


def calculate_tracking_error(
    self: "FeedbackPipeline",
    planned_state: np.ndarray,
    sim_qpos: np.ndarray,
    sim_qvel: np.ndarray,
) -> float:
    """Calculate tracking error between planned and simulated trajectories."""
    try:
        # Convert simulation results to MPC state format
        sim_states = []
        for i in range(min(len(sim_qpos), planned_state.shape[0])):
            sim_state, _ = conversion.sim_to_mpc(sim_qpos[i], sim_qvel[i])
            sim_states.append(sim_state)

        if not sim_states:
            return float("inf")

        sim_state_traj = np.array(sim_states)

        # Calculate RMS error for comparable lengths
        min_length = min(planned_state.shape[0], sim_state_traj.shape[0])
        planned_truncated = planned_state[:min_length]
        sim_truncated = sim_state_traj[:min_length]

        # Focus on key state components (position, orientation)
        key_states = np.concatenate(
            [
                planned_truncated[:, 0:3],  # COM position
                planned_truncated[:, 6:9],  # Euler angles
            ],
            axis=1,
        )

        sim_key_states = np.concatenate(
            [
                sim_truncated[:, 0:3],
                sim_truncated[:, 6:9],
            ],
            axis=1,
        )

        error = np.sqrt(np.mean((key_states - sim_key_states) ** 2))
        return float(error)

    except Exception as e:
        print(f"Error calculating tracking error: {e}")
        return float("inf")


def analyze_simulation_quality(
    self: "FeedbackPipeline",
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    grf_traj: np.ndarray,
    tracking_error: float,
) -> dict[str, Any]:
    """Analyze the quality and realism of the simulation."""

    analysis: dict[str, Any] = {
        "realistic": True,
        "max_joint_velocity": 0.0,
        "max_grf": 0.0,
        "trajectory_length": len(qpos_traj),
        "issues": [],
    }

    try:
        # Check for realistic joint velocities
        if len(qvel_traj) > 0:
            joint_velocities = qvel_traj[:, 6:]  # Skip base velocities
            max_joint_vel = np.max(np.abs(joint_velocities))
            analysis["max_joint_velocity"] = float(max_joint_vel)

            if max_joint_vel > 20.0:  # Unrealistic joint velocity
                analysis["realistic"] = False
                analysis["issues"].append("Unrealistic joint velocities")

        # Check ground reaction forces
        if len(grf_traj) > 0:
            max_grf = np.max(np.abs(grf_traj))
            analysis["max_grf"] = float(max_grf)

            if max_grf > 1000.0:  # Unrealistic forces
                analysis["realistic"] = False
                analysis["issues"].append("Unrealistic ground reaction forces")

        # Check tracking error
        if tracking_error > 0.5:
            analysis["realistic"] = False
            analysis["issues"].append("Poor trajectory tracking")

        # Check for simulation failure
        if len(qpos_traj) < 10:  # Very short trajectory suggests failure
            analysis["realistic"] = False
            analysis["issues"].append("Simulation terminated early")

    except Exception as e:
        analysis["realistic"] = False
        analysis["issues"].append(f"Analysis failed: {str(e)}")

    return analysis
