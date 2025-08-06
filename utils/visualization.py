import numpy as np


def euler_to_quaternion(
    rolls: np.ndarray, pitches: np.ndarray, yaws: np.ndarray
) -> np.ndarray:
    """
    Converts batched roll, pitch, and yaw Euler angles to quaternions.
    The order of rotation is ZYX (yaw, pitch, roll).

    Args:
        rolls (np.ndarray): A 1D numpy array of roll angles in radians.
        pitches (np.ndarray): A 1D numpy array of pitch angles in radians.
        yaws (np.ndarray): A 1D numpy array of yaw angles in radians.

    Returns:
        np.ndarray: A 2D numpy array of shape (N, 4) where N is the batch
                    size. Each row represents a quaternion in the format
                    [w, x, y, z].
    """
    cr = np.cos(rolls * 0.5)
    sr = np.sin(rolls * 0.5)
    cp = np.cos(pitches * 0.5)
    sp = np.sin(pitches * 0.5)
    cy = np.cos(yaws * 0.5)
    sy = np.sin(yaws * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.stack([w, x, y, z], axis=-1)


def render_planned_trajectory(
    state_traj: np.ndarray, joint_vel_traj: np.ndarray, env
) -> None:
    """
    Renders the planned trajectory.
    Args:
        state_traj: (N, 12)
        joint_vel_traj: (N, 12)
        env: gym.Env
    """
    qpos = np.zeros((state_traj.shape[0], 19))
    qvel = np.zeros((state_traj.shape[0], 18))

    # TODO: the position here is off as I am plotting the CoM position/velocity
    # instead of the base position/velocity
    qpos[:, 0:3] = state_traj[:, 0:3]
    qpos[:, 3:7] = euler_to_quaternion(
        state_traj[:, 6], state_traj[:, 7], state_traj[:, 8]
    )
    qpos[:, 7:] = state_traj[:, 12:24]
    qvel[:, 0:3] = state_traj[:, 3:6]
    qvel[:, 3:6] = state_traj[:, 9:12]
    qvel[:-1, 6:] = joint_vel_traj

    images = []
    for i in range(state_traj.shape[0]):
        env.reset(qpos=qpos[i], qvel=qvel[i])
        images.append(env.render(mode="rgb_array"))
    return images
