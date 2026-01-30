import numpy as np

QPOS_BASE_POS = slice(0, 3)
QPOS_BASE_QUAT = slice(3, 7)
QPOS_JOINTS = slice(7, 19)

QVEL_BASE_LIN = slice(0, 3)
QVEL_BASE_ANG = slice(3, 6)
QVEL_JOINTS = slice(6, 18)

MPC_X_BASE_POS = slice(0, 3)
MPC_X_BASE_VEL = slice(3, 6)
MPC_X_BASE_EUL = slice(6, 9)
MPC_X_BASE_ANG = slice(9, 12)
MPC_X_Q_JOINTS = slice(12, 24)
MPC_X_INTEGRAL = slice(24, 30)

MPC_U_QVEL_JOINTS = slice(0, 12)
MPC_U_GRF = slice(12, 24)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to Euler angles.
    The order of rotation is ZYX (yaw, pitch, roll).
    """
    qw, qx, qy, qz = q
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(2 * (qw * qy - qz * qx))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return np.array([roll, pitch, yaw])


def sim_to_mpc(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """
    Converts the qpos and qvel in the simulation format to the MPC state and input.
    """
    mpc_x = np.concatenate(
        [
            qpos[QPOS_BASE_POS],
            qvel[QVEL_BASE_LIN],
            quaternion_to_euler(qpos[QPOS_BASE_QUAT]),
            qvel[QVEL_BASE_ANG],
            qpos[QPOS_JOINTS],
            np.zeros(6),
        ]
    )
    mpc_u = np.concatenate(
        [
            qvel[QVEL_JOINTS],
            np.zeros(12),
        ]
    )
    return mpc_x, mpc_u


def mpc_to_sim(mpc_x: np.ndarray, mpc_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts the MPC state and input to the simulation format.
    """
    qpos = np.concatenate(
        [
            mpc_x[MPC_X_BASE_POS],
            euler_to_quaternion(mpc_x[MPC_X_BASE_EUL]),
            mpc_x[MPC_X_Q_JOINTS],
        ]
    )
    qvel = np.concatenate(
        [
            mpc_x[MPC_X_BASE_VEL],
            mpc_x[MPC_X_BASE_ANG],
            mpc_u[MPC_U_QVEL_JOINTS],
        ]
    )
    return qpos, qvel
