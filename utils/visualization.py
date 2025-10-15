import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv

from .conversion import mpc_to_sim


def render_planned_trajectory(
    state_traj: np.ndarray, joint_vel_traj: np.ndarray, env: QuadrupedEnv
) -> list[np.ndarray]:
    """
    Renders the planned trajectory.
    Args:
        state_traj: (N, 12)
        joint_vel_traj: (N, 12)
        env: gym.Env
    """

    images = []
    for i in range(state_traj.shape[0]):
        state = state_traj[i]
        joint_vel = (
            joint_vel_traj[i] if i < joint_vel_traj.shape[0] else joint_vel_traj[-1]
        )
        qpos, qvel = mpc_to_sim(state, joint_vel)
        env.reset(qpos=qpos, qvel=qvel)
        images.append(env.render(mode="rgb_array"))
    return images
