from typing import Any

import casadi as cs
import numpy as np
from liecasadi import SO3

# ============================================================
# Standardized State Indexing Constants (like forward dynamics)
# ============================================================
QP_BASE_POS = slice(0, 3)
QP_BASE_QUAT = slice(3, 7)
QP_JOINTS = slice(7, 19)

QV_BASE_LIN = slice(0, 3)
QV_BASE_ANG = slice(3, 6)
QV_JOINTS = slice(6, 18)

MP_X_BASE_POS = slice(0, 3)
MP_X_BASE_VEL = slice(3, 6)
MP_X_BASE_EUL = slice(6, 9)
MP_X_BASE_ANG = slice(9, 12)
MP_X_Q = slice(12, 24)

MP_U_QD = slice(0, 12)
MP_U_CONTACT_F = slice(12, 24)


# ============================================================
# Utilities for state conversions and FK
# ============================================================
def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to rotation matrix"""
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R


def euler_xyz_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles [roll,pitch,yaw] to quaternion [w,x,y,z]"""
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


# ============================================================
# Forward Kinematics for Foot Position Validation
# ============================================================
HIP_OFFSET_X = 0.183  # forward/back from COM
HIP_OFFSET_Y = 0.0838  # lateral offset
HIP_LINK = 0.2
UPPER_LINK = 0.2
LOWER_LINK = 0.2

LEG_ORDER = ["FL", "FR", "RL", "RR"]
LEG_OFFSETS = {
    "FL": np.array([HIP_OFFSET_X, HIP_OFFSET_Y, 0.0]),
    "FR": np.array([HIP_OFFSET_X, -HIP_OFFSET_Y, 0.0]),
    "RL": np.array([-HIP_OFFSET_X, HIP_OFFSET_Y, 0.0]),
    "RR": np.array([-HIP_OFFSET_X, -HIP_OFFSET_Y, 0.0]),
}


def foot_fk_local(joints: np.ndarray) -> np.ndarray:
    """Return foot position in base frame for given [abd, hip, knee]"""
    abd, hip, knee = joints
    # Lateral offset from abduction
    y = HIP_OFFSET_Y * np.sign(abd if abd != 0 else 1)
    # Sagittal plane
    x = HIP_LINK * np.cos(hip)
    z = -HIP_LINK * np.sin(hip)
    x += UPPER_LINK * np.cos(hip + knee)
    z += -UPPER_LINK * np.sin(hip + knee)
    x += LOWER_LINK * np.cos(hip + knee)
    z += -LOWER_LINK * np.sin(hip + knee)
    return np.array([0.0, y, 0.0]) + np.array([x, 0, z])


def feet_positions_world_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """Compute world foot positions from qpos using standardized indexing"""
    base_pos = qpos[QP_BASE_POS]
    base_quat = qpos[QP_BASE_QUAT]
    R = quat_wxyz_to_rotmat(base_quat)

    feet = []
    for i, leg in enumerate(LEG_ORDER):
        joints = qpos[QP_JOINTS][3 * i : 3 * (i + 1)]
        p_local = LEG_OFFSETS[leg] + foot_fk_local(joints)
        p_world = base_pos + R @ p_local
        feet.append(p_world)
    return np.array(feet)


# ============================================================
# Improved Numerical Methods for Acceleration
# ============================================================
def compute_accelerations_improved(
    state_traj: np.ndarray,
    input_traj: np.ndarray,
    contact_traj: np.ndarray,
    dt: float,
    kinodynamic_model: Any,
) -> np.ndarray:
    """Compute accelerations using forward dynamics for better stability"""
    import config

    num_steps = len(state_traj) - 1
    ddq_traj = np.zeros((num_steps, 18))

    # Forward dynamics parameters template
    param_template = np.concatenate(
        [
            np.ones(4),  # contact state
            np.array([config.experiment.mu_ground]),
            np.zeros(4),  # stance proximity
            np.zeros(3),  # base position
            np.array([0.0]),  # base yaw
            np.zeros(6),  # external wrench
            config.robot_data.inertia.flatten(),
            np.array([config.robot_data.mass]),
        ]
    )

    for i in range(num_steps):
        # Use forward dynamics to get accurate accelerations
        param = param_template.copy()
        param[0:4] = contact_traj[:, i].astype(float)
        param[9:12] = state_traj[i][MP_X_BASE_POS]
        param[12] = state_traj[i][MP_X_BASE_EUL][2]  # yaw

        # Get state derivative from forward dynamics
        xdot = kinodynamic_model.forward_dynamics(
            state_traj[i][:, None], input_traj[i][:, None], param[:, None]
        )
        xdot_array = cs.DM(xdot).toarray()[:, 0]

        # Extract accelerations from state derivative
        ddq_traj[i, QV_BASE_LIN] = xdot_array[MP_X_BASE_VEL]
        ddq_traj[i, QV_BASE_ANG] = xdot_array[MP_X_BASE_ANG]

        # Joint accelerations from central difference for stability
        if i < num_steps - 1:
            ddq_traj[i, QV_JOINTS] = (
                input_traj[i + 1][MP_U_QD] - input_traj[i][MP_U_QD]
            ) / dt
        else:
            ddq_traj[i, QV_JOINTS] = ddq_traj[i - 1, QV_JOINTS]  # extrapolate

    return ddq_traj


def compute_joint_torques(
    kindyn_model: Any,
    state_traj: np.ndarray,
    input_traj: np.ndarray,
    contact_sequence: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Improved inverse dynamics computation with systematic improvements:
    - Standardized state indexing using QP_*, QV_*, MP_* constants
    - Forward kinematics for foot position validation
    - Improved numerical methods for acceleration computation
    - Ground contact validation using FK-based foot positions

    This function implements the inverse dynamics equation:
    tau = M(q) * ddq + C(q, dq) + g(q) - J_contact^T * F_contact

    Args:
        kindyn_model (KinoDynamic_Model): An instance of the kinodynamic model
            which contains the KinDynComputations object from the 'adam' library.
        state_traj (np.ndarray): The full state trajectory from the optimizer.
            Shape: (num_steps + 1, num_states).
        input_traj (np.ndarray): The input trajectory of the kinodynamic model (qvel, grf)
            Shape: (num_steps, 24).
        contact_sequence (np.ndarray): Array of contact flags (1 for stance, 0 for swing).
            Shape: (4, num_steps) - NOTE: Transposed from old version.
        dt (float): The time step duration.

    Returns:
        np.ndarray: The computed joint torque trajectory.
        Shape: (num_steps, 12).
    """
    # Initialize output array
    num_steps = input_traj.shape[0]
    num_joints = 12  # 4 legs * 3 joints
    joint_torques_traj = np.zeros((num_steps, num_joints))

    print("Building state representations with standardized indexing...")

    # Build full qpos trajectory using standardized indexing
    q_traj = np.zeros((num_steps + 1, 19))
    q_traj[:, QP_BASE_POS] = state_traj[:, MP_X_BASE_POS]

    # Convert RPY to Quaternion using robust method
    for i in range(num_steps + 1):
        euler = state_traj[i, MP_X_BASE_EUL]
        quat_wxyz = euler_xyz_to_quat_wxyz(euler)
        q_traj[i, QP_BASE_QUAT] = quat_wxyz

    q_traj[:, QP_JOINTS] = state_traj[:, MP_X_Q]

    # Build full qvel trajectory using standardized indexing
    dq_traj = np.zeros((num_steps + 1, 18))
    dq_traj[:, QV_BASE_LIN] = state_traj[:, MP_X_BASE_VEL]
    dq_traj[:, QV_BASE_ANG] = state_traj[:, MP_X_BASE_ANG]
    dq_traj[:-1, QV_JOINTS] = input_traj[:, MP_U_QD]
    dq_traj[-1, QV_JOINTS] = input_traj[-1, MP_U_QD]  # extend for final step

    print("Computing accelerations using improved numerical methods...")
    # Use improved acceleration computation instead of naive finite differences
    ddq_traj = compute_accelerations_improved(
        state_traj, input_traj, contact_sequence, dt, kindyn_model
    )

    # Extend for final step
    ddq_traj_extended = np.zeros((num_steps + 1, 18))
    ddq_traj_extended[:-1] = ddq_traj
    ddq_traj_extended[-1] = ddq_traj[-1]  # extrapolate

    print("Setting up symbolic inverse dynamics with proper structure...")

    # Symbolic variables for CasADi function generation
    base_pos_sym = cs.SX.sym("base_pos", 3)
    base_quat_sym = cs.SX.sym("base_quat", 4)  # x,y,z,w format
    joint_pos_sym = cs.SX.sym("joint_pos", 12)
    base_vel_sym = cs.SX.sym("base_vel", 6)
    joint_vel_sym = cs.SX.sym("joint_vel", 12)
    base_acc_sym = cs.SX.sym("base_acc", 6)
    joint_acc_sym = cs.SX.sym("joint_acc", 12)
    f_ext_sym = cs.SX.sym("f_ext", 12)

    # Construct homogeneous transformation matrix H from quaternion
    # Adam library expects quaternion in [w, x, y, z] order
    quat_wxyz_sym = cs.vertcat(base_quat_sym[3], base_quat_sym[0:3])
    H = cs.SX.eye(4)
    H[0:3, 0:3] = SO3.from_quat(quat_wxyz_sym).as_matrix()
    H[0:3, 3] = base_pos_sym

    # Create symbolic functions from kinodynamic model
    mass_matrix_fun = kindyn_model.kindyn.mass_matrix_fun()
    bias_force_fun = kindyn_model.kindyn.bias_force_fun()
    gravity_fun = kindyn_model.kindyn.gravity_term_fun()

    J_FL_fun = kindyn_model.kindyn.jacobian_fun("FL_foot")
    J_FR_fun = kindyn_model.kindyn.jacobian_fun("FR_foot")
    J_RL_fun = kindyn_model.kindyn.jacobian_fun("RL_foot")
    J_RR_fun = kindyn_model.kindyn.jacobian_fun("RR_foot")

    # Symbolic inverse dynamics computation
    M_sym = mass_matrix_fun(H, joint_pos_sym)
    C_sym = bias_force_fun(H, joint_pos_sym, base_vel_sym, joint_vel_sym)
    g_sym = gravity_fun(H, joint_pos_sym)

    # Split external forces by leg
    F_FL, F_FR, F_RL, F_RR = cs.vertsplit(f_ext_sym, 3)
    J_FL, J_FR, J_RL, J_RR = (
        f(H, joint_pos_sym)[0:3, :] for f in [J_FL_fun, J_FR_fun, J_RL_fun, J_RR_fun]
    )
    wrench_ext = J_FL.T @ F_FL + J_FR.T @ F_FR + J_RL.T @ F_RL + J_RR.T @ F_RR

    ddq_full_sym = cs.vertcat(base_acc_sym, joint_acc_sym)

    # Correct inverse dynamics equation: tau = M*ddq + C + g - J^T*F
    tau_full_sym = M_sym @ ddq_full_sym + C_sym + g_sym - wrench_ext

    # Create CasADi function
    inverse_dynamics_fun = cs.Function(
        "inverse_dynamics",
        [
            base_pos_sym,
            base_quat_sym,
            joint_pos_sym,
            base_vel_sym,
            joint_vel_sym,
            base_acc_sym,
            joint_acc_sym,
            f_ext_sym,
        ],
        [tau_full_sym],
    )

    print("Evaluating inverse dynamics with contact validation...")

    # Evaluation loop with foot position validation
    grf_traj = input_traj[:, MP_U_CONTACT_F]

    for i in range(num_steps):
        # Apply contact forces only for legs in contact using standardized indexing
        grfs_vec = grf_traj[i, :].copy()
        for leg_idx in range(4):
            contact_state = contact_sequence[leg_idx, i]
            grfs_vec[3 * leg_idx : 3 * (leg_idx + 1)] *= contact_state

            # Validate foot position for contact (ground contact validation)
            feet_world = feet_positions_world_from_qpos(q_traj[i])
            foot_height = feet_world[leg_idx, 2]

            # Issue warnings for contact/foot position mismatches
            if (
                contact_state == 1 and foot_height > 0.02
            ):  # Contact claimed but foot in air
                if i % 50 == 0:  # Limit warning frequency
                    print(
                        f"Warning: Step {i}, Leg {leg_idx} contact mismatch - foot at {foot_height:.3f}m"
                    )
            elif (
                contact_state == 0 and foot_height < -0.01
            ):  # No contact but foot underground
                if i % 50 == 0:
                    print(
                        f"Warning: Step {i}, Leg {leg_idx} penetration - foot at {foot_height:.3f}m"
                    )

        # Compute inverse dynamics using standardized indexing
        tau_full = inverse_dynamics_fun(
            q_traj[i, QP_BASE_POS],
            q_traj[i, QP_BASE_QUAT],
            q_traj[i, QP_JOINTS],
            np.concatenate(
                [dq_traj[i, QV_BASE_LIN], dq_traj[i, QV_BASE_ANG]]
            ),  # Concatenate base lin+ang vel
            dq_traj[i, QV_JOINTS],
            np.concatenate(
                [ddq_traj_extended[i, QV_BASE_LIN], ddq_traj_extended[i, QV_BASE_ANG]]
            ),  # Concatenate base lin+ang acc
            ddq_traj_extended[i, QV_JOINTS],
            grfs_vec,
        )

        # Extract joint torques (skip the first 6 base wrench elements)
        joint_torques_traj[i, :] = tau_full.full().flatten()[6:]

    print("Inverse dynamics computation completed!")
    return joint_torques_traj


def compute_joint_torques_original(
    kindyn_model: Any,
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Original inverse dynamics computation using finite differences

    This is the original implementation that:
    - Uses finite differences to compute velocities and accelerations
    - Uses manual state indexing
    - Takes only GRF trajectory as input (not optimized joint velocities)

    Kept for comparison with the improved implementation.

    Args:
        kindyn_model (KinoDynamic_Model): An instance of the kinodynamic model
        state_traj (np.ndarray): The full state trajectory from the optimizer
        grf_traj (np.ndarray): The GRF trajectory
        contact_sequence (np.ndarray): Array of contact flags
        dt (float): The time step duration

    Returns:
        np.ndarray: The computed joint torque trajectory
    """
    # Initialize output
    num_steps = grf_traj.shape[0]
    joint_torques_traj = np.zeros((num_steps, 12))

    print("Building state representations with original manual indexing...")

    # Build position trajectory using original manual indexing
    pos_kindyn_traj = state_traj[:, [0, 1, 2, 6, 7, 8]]  # Base pos + RPY
    pos_kindyn_traj = np.hstack([pos_kindyn_traj, state_traj[:, 12:24]])  # Add joints

    # Compute velocities using finite differences (original method)
    dq_traj = np.zeros((num_steps + 1, pos_kindyn_traj.shape[1]))
    for i in range(num_steps):
        dq_traj[i, :] = (pos_kindyn_traj[i + 1, :] - pos_kindyn_traj[i, :]) / dt
    dq_traj[-1, :] = dq_traj[-2, :]  # Extrapolate last step

    # Compute accelerations using finite differences (original method)
    ddq_traj = np.zeros((num_steps + 1, pos_kindyn_traj.shape[1]))
    for i in range(num_steps):
        ddq_traj[i, :] = (dq_traj[i + 1, :] - dq_traj[i, :]) / dt
    ddq_traj[-1, :] = ddq_traj[-2, :]  # Extrapolate last step

    print("Computing velocities and accelerations using finite differences...")

    # Convert to full qpos trajectory (19 DOF: 3 pos + 4 quat + 12 joints)
    q_traj = np.zeros((num_steps + 1, 19))
    q_traj[:, 0:3] = pos_kindyn_traj[:, 0:3]  # Base position

    # Convert RPY to quaternion
    for i in range(num_steps + 1):
        euler = pos_kindyn_traj[i, 3:6]
        quat_wxyz = euler_xyz_to_quat_wxyz(euler)
        q_traj[i, 3:7] = quat_wxyz

    q_traj[:, 7:19] = pos_kindyn_traj[:, 6:18]  # Joint positions

    # Build velocity trajectory (18 DOF: 6 base + 12 joints)
    dq_full_traj = np.zeros((num_steps + 1, 18))
    dq_full_traj[:, 0:3] = dq_traj[:, 0:3]  # Base linear velocity
    dq_full_traj[:, 3:6] = dq_traj[:, 3:6]  # Base angular velocity (RPY rates)
    dq_full_traj[:, 6:18] = dq_traj[:, 6:18]  # Joint velocities

    # Build acceleration trajectory
    ddq_full_traj = np.zeros((num_steps + 1, 18))
    ddq_full_traj[:, 0:3] = ddq_traj[:, 0:3]  # Base linear acceleration
    ddq_full_traj[:, 3:6] = ddq_traj[:, 3:6]  # Base angular acceleration
    ddq_full_traj[:, 6:18] = ddq_traj[:, 6:18]  # Joint accelerations

    print("Setting up symbolic inverse dynamics...")

    # Set up symbolic inverse dynamics (similar to improved version)
    base_pos_sym = cs.SX.sym("base_pos", 3)
    base_quat_sym = cs.SX.sym("base_quat", 4)  # x,y,z,w format
    joint_pos_sym = cs.SX.sym("joint_pos", 12)
    base_vel_sym = cs.SX.sym("base_vel", 6)
    joint_vel_sym = cs.SX.sym("joint_vel", 12)
    base_acc_sym = cs.SX.sym("base_acc", 6)
    joint_acc_sym = cs.SX.sym("joint_acc", 12)
    f_ext_sym = cs.SX.sym("f_ext", 12)

    # Construct homogeneous transformation matrix
    quat_wxyz_sym = cs.vertcat(base_quat_sym[3], base_quat_sym[0:3])
    H = cs.SX.eye(4)
    H[0:3, 0:3] = SO3.from_quat(quat_wxyz_sym).as_matrix()
    H[0:3, 3] = base_pos_sym

    # Create symbolic functions
    mass_matrix_fun = kindyn_model.kindyn.mass_matrix_fun()
    bias_force_fun = kindyn_model.kindyn.bias_force_fun()
    gravity_fun = kindyn_model.kindyn.gravity_term_fun()

    J_FL_fun = kindyn_model.kindyn.jacobian_fun("FL_foot")
    J_FR_fun = kindyn_model.kindyn.jacobian_fun("FR_foot")
    J_RL_fun = kindyn_model.kindyn.jacobian_fun("RL_foot")
    J_RR_fun = kindyn_model.kindyn.jacobian_fun("RR_foot")

    # Symbolic inverse dynamics computation
    M_sym = mass_matrix_fun(H, joint_pos_sym)
    C_sym = bias_force_fun(H, joint_pos_sym, base_vel_sym, joint_vel_sym)
    g_sym = gravity_fun(H, joint_pos_sym)

    # Split external forces by leg
    F_FL, F_FR, F_RL, F_RR = cs.vertsplit(f_ext_sym, 3)
    J_FL, J_FR, J_RL, J_RR = (
        f(H, joint_pos_sym)[0:3, :] for f in [J_FL_fun, J_FR_fun, J_RL_fun, J_RR_fun]
    )
    wrench_ext = J_FL.T @ F_FL + J_FR.T @ F_FR + J_RL.T @ F_RL + J_RR.T @ F_RR

    ddq_full_sym = cs.vertcat(base_acc_sym, joint_acc_sym)

    # Inverse dynamics equation
    tau_full_sym = M_sym @ ddq_full_sym + C_sym + g_sym - wrench_ext

    # Create CasADi function
    inverse_dynamics_fun = cs.Function(
        "inverse_dynamics",
        [
            base_pos_sym,
            base_quat_sym,
            joint_pos_sym,
            base_vel_sym,
            joint_vel_sym,
            base_acc_sym,
            joint_acc_sym,
            f_ext_sym,
        ],
        [tau_full_sym],
    )

    print("Evaluating inverse dynamics...")

    # Evaluation loop
    for i in range(num_steps):
        # Apply contact forces
        grfs_vec = grf_traj[i, :].copy()
        for leg_idx in range(4):
            contact_state = contact_sequence[leg_idx, i]
            grfs_vec[3 * leg_idx : 3 * (leg_idx + 1)] *= contact_state

        # Compute inverse dynamics
        tau_full = inverse_dynamics_fun(
            q_traj[i, 0:3],  # base position
            q_traj[i, 3:7],  # base quaternion [w,x,y,z]
            q_traj[i, 7:19],  # joint positions
            dq_full_traj[i, 0:6],  # base velocity [lin, ang]
            dq_full_traj[i, 6:18],  # joint velocities
            ddq_full_traj[i, 0:6],  # base acceleration [lin, ang]
            ddq_full_traj[i, 6:18],  # joint accelerations
            grfs_vec,
        )

        # Extract joint torques (skip first 6 base wrench elements)
        joint_torques_traj[i, :] = tau_full.full().flatten()[6:]

    print("Original inverse dynamics computation completed!")
    return joint_torques_traj
