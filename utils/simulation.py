"""
Simulation utilities for trajectory execution and data collection.
"""

from __future__ import annotations

import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from tqdm import tqdm

# TODO: remove this dependency on config; directly pass the parameters to the function
import config


def create_reference_trajectory(
    initial_qpos: np.ndarray, target_jump_height: float = 0.15
) -> np.ndarray:
    """
    Create reference trajectory for MPC optimization.

    Args:
        initial_qpos: Initial joint positions from simulation
        target_jump_height: Target jump height in meters

    Returns:
        Reference trajectory vector for MPC optimization
    """
    reference = {
        "ref_position": initial_qpos[0:3] + np.array([0.1, 0.0, target_jump_height]),
        "ref_linear_velocity": np.array([0.0, 0.0, 0.0]),
        "ref_orientation": np.zeros(3),
        "ref_angular_velocity": np.zeros(3),
        "ref_joints": initial_qpos[7:19],
    }

    return np.concatenate(
        [
            reference["ref_position"],
            reference["ref_linear_velocity"],
            reference["ref_orientation"],
            reference["ref_angular_velocity"],
            reference["ref_joints"],
            np.zeros(6),  # Reference for integral states
            np.zeros(24),  # Reference for inputs
        ]
    )


def save_trajectory_results(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray,
    suffix: str = "",
) -> None:
    """
    Save all trajectory optimization results to files.

    Args:
        state_traj: State trajectory from MPC optimization
        joint_vel_traj: Joint velocity trajectory
        grf_traj: Ground reaction force trajectory
        contact_sequence: Contact sequence matrix
        suffix: File suffix for saving
    """
    np.save(f"results/state_traj{suffix}.npy", state_traj)
    np.save(f"results/joint_vel_traj{suffix}.npy", joint_vel_traj)
    np.save(f"results/grf_traj{suffix}.npy", grf_traj)
    np.save("results/contact_sequence.npy", contact_sequence)


def simulate_trajectory(
    env: QuadrupedEnv,
    joint_torques_traj: np.ndarray,
    planned_traj_images: list[np.ndarray] | None = None,
    suffix: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Simulate the computed trajectory and optionally render/save video.

    Args:
        env: Quadruped simulation environment
        joint_torques_traj: Joint torque trajectory to execute
        planned_traj_images: Optional planned trajectory images for overlay
        suffix: File suffix for saving results

    Returns:
        Tuple of (qpos_traj, qvel_traj, grf_traj, images) -
        position, velocity, ground reaction force, and rendered images
    """
    env.reset(qpos=config.experiment.initial_qpos, qvel=config.experiment.initial_qvel)

    if not config.experiment.render:
        return _run_simulation_without_rendering(env, joint_torques_traj)
    else:
        return _run_simulation_with_rendering(
            env, joint_torques_traj, planned_traj_images, suffix
        )


def _run_simulation_without_rendering(
    env: QuadrupedEnv, joint_torques_traj: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Run simulation without rendering for performance.

    Args:
        env: Quadruped simulation environment
        joint_torques_traj: Joint torque trajectory to execute

    Returns:
        Tuple of (qpos_traj, qvel_traj, images) - position, velocity trajectories, and empty images list
    """
    num_steps = int(config.experiment.duration / config.experiment.sim_dt)
    qpos_traj = []
    qvel_traj = []
    grf_traj = []
    action_index = 0

    for i in tqdm(range(num_steps)):
        action = joint_torques_traj[action_index, :]
        if (i + 1) % int(config.mpc_config.mpc_dt / config.experiment.sim_dt) == 0:
            action_index += 1

        state, reward, is_terminated, is_truncated, info = env.step(action=action)
        qpos_traj.append(state["qpos"].copy())
        qvel_traj.append(state["qvel"].copy())
        grf_traj.append(state["contact_forces:base"].copy())
    return np.array(qpos_traj), np.array(qvel_traj), np.array(grf_traj), []


def _run_simulation_with_rendering(
    env: QuadrupedEnv,
    joint_torques_traj: np.ndarray,
    planned_traj_images: list[np.ndarray] | None,
    suffix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Run simulation with rendering and video generation.

    Args:
        env: Quadruped simulation environment
        joint_torques_traj: Joint torque trajectory to execute
        planned_traj_images: Planned trajectory images for overlay
        suffix: File suffix for saving video

    Returns:
        Tuple of (qpos_traj, qvel_traj, images) - position, velocity trajectories, and rendered images
    """

    num_steps = int(config.experiment.duration / config.experiment.sim_dt)
    images = []
    qpos_traj = []
    qvel_traj = []
    grf_traj = []
    action_index = 0

    for i in tqdm(range(num_steps)):
        action = joint_torques_traj[action_index, :]
        if (i + 1) % int(config.mpc_config.mpc_dt / config.experiment.sim_dt) == 0:
            action_index += 1

        state, reward, is_terminated, is_truncated, info = env.step(action=action)
        qpos_traj.append(state["qpos"].copy())
        qvel_traj.append(state["qvel"].copy())
        grf_traj.append(state["contact_forces:base"].copy())
        # Render and composite images
        image = env.render(mode="rgb_array", tint_robot=True)
        overplotted_image = np.uint8(
            0.7 * image
            + 0.3
            * (
                planned_traj_images[action_index]
                if planned_traj_images is not None
                else image
            )
        )
        images.append(overplotted_image)

    return np.array(qpos_traj), np.array(qvel_traj), np.array(grf_traj), images  # type: ignore[return-value]
