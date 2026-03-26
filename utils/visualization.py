from __future__ import annotations

import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv

from .conversion import mpc_to_sim


def render_planned_trajectory(
    state_traj: np.ndarray, joint_vel_traj: np.ndarray, env: QuadrupedEnv
) -> list[np.ndarray]:
    """
    Renders the planned trajectory using offscreen mujoco.Renderer.
    Args:
        state_traj: (N, 12)
        joint_vel_traj: (N, 12)
        env: gym.Env
    """
    import mujoco

    renderer = mujoco.Renderer(env.mjModel, height=480, width=640)
    images = []
    for i in range(state_traj.shape[0]):
        state = state_traj[i]
        joint_vel = (
            joint_vel_traj[i] if i < joint_vel_traj.shape[0] else joint_vel_traj[-1]
        )
        qpos, qvel = mpc_to_sim(state, joint_vel)
        env.reset(qpos=qpos, qvel=qvel)
        renderer.update_scene(env.mjData)
        images.append(renderer.render())
    renderer.close()
    return images


def render_and_save_planned_trajectory(
    state_traj: np.ndarray,
    input_traj: np.ndarray,
    env: QuadrupedEnv,
    suffix: str = "",
    fps: float | None = None,
) -> list[np.ndarray] | None:
    """
    Render planned trajectory and save as video if rendering is enabled.

    Args:
        state_traj: State trajectory from MPC
        input_traj: Input trajectory (joint velocities + forces)
        env: Quadruped simulation environment
        suffix: File suffix for saving
        fps: Frames per second for video (if None, uses MPC dt)

    Returns:
        List of rendered images if rendering is enabled, None otherwise
    """
    import imageio

    import go2_config as config

    if not config.experiment.render:
        return None

    print("Rendering planned trajectory...")
    joint_vel_traj = input_traj[:, :12]  # Extract joint velocities
    planned_traj_images = render_planned_trajectory(state_traj, joint_vel_traj, env)

    if fps is None:
        fps = 1 / config.mpc_config.mpc_dt

    imageio.mimsave(
        f"results/planned_traj{suffix}.mp4",
        planned_traj_images,
        fps=fps,
    )
    return planned_traj_images
