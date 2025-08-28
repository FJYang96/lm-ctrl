import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from liecasadi import SO3


def compute_joint_torques(
    kindyn_model,
    state_traj: np.ndarray,
    input_traj: np.ndarray,
    contact_sequence: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Computes joint torques for the full robot from a given state and GRF trajectory.

    This function implements the inverse dynamics equation:
    tau = M(q) * ddq + C(q, dq) * dq - J_contact^T * F_contact

    Args:
        kindyn_model (KinoDynamic_Model): An instance of the kinodynamic model
            which contains the KinDynComputations object from the 'adam' library.
        state_traj (np.ndarray): The full state trajectory from the optimizer.
            Shape: (num_steps + 1, num_states).
        input_traj (np.ndarray): The input trajectory of the kinodynamic model (qvel, grf)
            Shape: (num_steps, 24).
        contact_sequence (np.ndarray): Array of contact flags (1 for stance, 0 for swing).
            Shape: (num_steps, 4).
        dt (float): The time step duration.

    Returns:
        np.ndarray: The computed joint torque trajectory.
        Shape: (num_steps, 12).
    """
    # Initialize joint_torques_traj, which will be returned
    num_steps = input_traj.shape[0]
    num_joints = 12  # 4 legs * 3 joints
    joint_torques_traj = np.zeros((num_steps, num_joints))

    # # Extract qpos, qvel from state_traj
    # # The state vector is:
    # # [CoM pos(3), CoM vel(3), RPY(3), angular vel(3), Joint pos(12), Integrals(6)]
    # qpos_traj = np.hstack(
    #     [state_traj[:, 0:3], state_traj[:, 6:9], state_traj[:, 12:24]]
    # )
    # qvel_traj, grf_traj = input_traj[:, :12], input_traj[:, 12:]
    # qvel_traj = np.hstack(
    #     [
    #         state_traj[:, 3:6],
    #         state_traj[:, 9:12],
    #         np.vstack((qvel_traj, qvel_traj[-1:])),
    #     ]
    # )

    # # We approximate the accelerations via finite differences
    # pos_kindyn_traj = qpos_traj
    # dq_traj = qvel_traj
    # ddq_traj = np.zeros_like(qvel_traj)
    # for i in range(num_steps - 1):
    #     ddq_traj[i, :] = (dq_traj[i + 1, :] - dq_traj[i, :]) / dt

    # # The KinDynComputations object needs symbolic variables for the function
    # # calls. We will create these symbolic variables once.
    # # The state is: [base_pos(3), base_rpy(3), joint_pos(12)] = 18 total
    # base_pos_sym = cs.SX.sym("base_pos", 3)
    # base_rpy_sym = cs.SX.sym("base_rpy", 3)
    # joint_pos_sym = cs.SX.sym("joint_pos", 12)
    # base_vel_sym = cs.SX.sym("base_vel", 6)  # linear + angular velocities
    # joint_vel_sym = cs.SX.sym("joint_vel", 12)
    # base_acc_sym = cs.SX.sym("base_acc", 6)  # linear + angular accelerations
    # joint_acc_sym = cs.SX.sym("joint_acc", 12)
    # f_ext_sym = cs.SX.sym("f_ext", 12)

    # # Extract roll, pitch, yaw from base_rpy_sym
    # roll, pitch, yaw = base_rpy_sym[0], base_rpy_sym[1], base_rpy_sym[2]

    # # Construct homogeneous transformation matrix H
    # w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    # b_R_w = w_R_b.T
    # H = cs.SX.eye(4)
    # H[0:3, 0:3] = b_R_w.T
    # H[0:3, 3] = base_pos_sym

    # mass_matrix_fun = kindyn_model.kindyn.mass_matrix_fun()
    # bias_force_fun = kindyn_model.kindyn.bias_force_fun()

    # # Get symbolic Jacobians for each foot
    # J_FL_fun = kindyn_model.kindyn.jacobian_fun("FL_foot")
    # J_FR_fun = kindyn_model.kindyn.jacobian_fun("FR_foot")
    # J_RL_fun = kindyn_model.kindyn.jacobian_fun("RL_foot")
    # J_RR_fun = kindyn_model.kindyn.jacobian_fun("RR_foot")

    # # Create the symbolic function for inverse dynamics
    # M_sym = mass_matrix_fun(H, joint_pos_sym)
    # C_sym = bias_force_fun(H, joint_pos_sym, base_vel_sym, joint_vel_sym)

    # F_FL_sym = f_ext_sym[0:3]
    # F_FR_sym = f_ext_sym[3:6]
    # F_RL_sym = f_ext_sym[6:9]
    # F_RR_sym = f_ext_sym[9:12]

    # J_FL_sym = J_FL_fun(H, joint_pos_sym)[0:3, :]
    # J_FR_sym = J_FR_fun(H, joint_pos_sym)[0:3, :]
    # J_RL_sym = J_RL_fun(H, joint_pos_sym)[0:3, :]
    # J_RR_sym = J_RR_fun(H, joint_pos_sym)[0:3, :]

    # wrench_ext = (
    #     J_FL_sym.T @ F_FL_sym
    #     + J_FR_sym.T @ F_FR_sym
    #     + J_RL_sym.T @ F_RL_sym
    #     + J_RR_sym.T @ F_RR_sym
    # )

    # # Combine accelerations: [base_acc(6), joint_acc(12)]
    # ddq_full_sym = cs.vertcat(base_acc_sym, joint_acc_sym)

    # tau_full_sym = M_sym @ ddq_full_sym + C_sym - wrench_ext

    # inverse_dynamics_fun = cs.Function(
    #     "inverse_dynamics",
    #     [
    #         base_pos_sym,
    #         base_rpy_sym,
    #         joint_pos_sym,
    #         base_vel_sym,
    #         joint_vel_sym,
    #         base_acc_sym,
    #         joint_acc_sym,
    #         f_ext_sym,
    #     ],
    #     [tau_full_sym],
    # )

    # for i in range(num_steps):
    #     # Extract components from trajectory
    #     # pos_kindyn_traj[i, :] = [base_pos(3), base_rpy(3), joint_pos(12)]
    #     base_pos_vec = pos_kindyn_traj[i, 0:3]
    #     base_rpy_vec = pos_kindyn_traj[i, 3:6]
    #     joint_pos_vec = pos_kindyn_traj[i, 6:18]

    #     # dq_traj[i, :] = [base_vel(3), base_rpy_vel(3), joint_vel(12)]
    #     # Note: For velocities, we need [linear_vel(3), angular_vel(3)] for base
    #     base_vel_vec = dq_traj[i, 0:6]  # [linear_vel(3), angular_vel(3)]
    #     joint_vel_vec = dq_traj[i, 6:18]

    #     # For the last step, acceleration is not available from finite differencing.
    #     # A common practice is to use the second-to-last acceleration.
    #     if i < num_steps - 1:
    #         base_acc_vec = ddq_traj[i, 0:6]
    #         joint_acc_vec = ddq_traj[i, 6:18]
    #     else:
    #         base_acc_vec = ddq_traj[num_steps - 2, 0:6]
    #         joint_acc_vec = ddq_traj[num_steps - 2, 6:18]

    #     # FIX 4: Correct indexing for contact_sequence with shape (4, num_steps).
    #     # Apply contact forces only for legs in stance.
    #     grfs_vec = grf_traj[i, :] * np.repeat(contact_sequence[:, i], 3)

    #     # Compute the full wrench (base and joints)
    #     tau_full = inverse_dynamics_fun(
    #         base_pos_vec,
    #         base_rpy_vec,
    #         joint_pos_vec,
    #         base_vel_vec,
    #         joint_vel_vec,
    #         base_acc_vec,
    #         joint_acc_vec,
    #         grfs_vec,
    #     )

    #     # We only need the joint torques (the last 12 values)
    #     joint_torques_traj[i, :] = tau_full.full().flatten()[6:]

    # return joint_torques_traj

    q_traj = np.zeros((num_steps + 1, 19))
    q_traj[:, 0:3] = state_traj[:, 0:3]  # Base position
    # Convert RPY to Quaternion for each step
    for i in range(num_steps + 1):
        rpy = state_traj[i, 6:9]
        # Using liecasadi for robust conversion, from_euler expects [roll, pitch, yaw]
        # Pinocchio/Adam use [x, y, z, w] convention
        quat_wxyz = SO3.from_euler(rpy).as_quat()
        q_traj[i, 3] = quat_wxyz.x
        q_traj[i, 4] = quat_wxyz.y
        q_traj[i, 5] = quat_wxyz.z
        q_traj[i, 6] = quat_wxyz.w
    q_traj[:, 7:19] = state_traj[:, 12:24]  # Joint positions

    # Assemble the full velocity vector dq
    dq_traj = np.zeros((num_steps + 1, 18))
    dq_traj[:, 0:3] = state_traj[:, 3:6]  # Base linear velocity
    dq_traj[:, 3:6] = state_traj[:, 9:12]  # Base angular velocity
    dq_traj[:-1, 6:18] = input_traj[:, :12]  # Joint velocities from input trajectory
    dq_traj[-1, 6:18] = input_traj[
        -1, :12
    ]  # Repeat last joint velocity for the final step

    # We approximate the accelerations via finite differences on the full velocity vector
    ddq_traj = np.zeros_like(dq_traj)
    for i in range(num_steps):
        ddq_traj[i, :] = (dq_traj[i + 1, :] - dq_traj[i, :]) / dt
    ddq_traj[-1, :] = ddq_traj[-2, :]  # Extrapolate for the last step

    grf_traj = input_traj[:, 12:]

    # Symbolic variables for CasADi function generation
    # These match the structure required by KinDynComputations
    base_pos_sym = cs.SX.sym("base_pos", 3)
    base_quat_sym = cs.SX.sym("base_quat", 4)  # x,y,z,w
    joint_pos_sym = cs.SX.sym("joint_pos", 12)

    base_vel_sym = cs.SX.sym("base_vel", 6)
    joint_vel_sym = cs.SX.sym("joint_vel", 12)

    base_acc_sym = cs.SX.sym("base_acc", 6)
    joint_acc_sym = cs.SX.sym("joint_acc", 12)
    f_ext_sym = cs.SX.sym("f_ext", 12)

    # Construct homogeneous transformation matrix H from quaternion
    # Adam library expects the quaternion in [w, x, y, z] order for H construction
    quat_wxyz_sym = cs.vertcat(base_quat_sym[3], base_quat_sym[0:3])
    H = cs.SX.eye(4)
    H[0:3, 0:3] = SO3.from_quat(quat_wxyz_sym).as_matrix()
    H[0:3, 3] = base_pos_sym

    # Create symbolic functions from the kinodynamic model
    mass_matrix_fun = kindyn_model.kindyn.mass_matrix_fun()
    bias_force_fun = kindyn_model.kindyn.bias_force_fun()
    gravity_fun = kindyn_model.kindyn.gravity_term_fun()
    J_FL_fun = kindyn_model.kindyn.jacobian_fun("FL_foot")
    J_FR_fun = kindyn_model.kindyn.jacobian_fun("FR_foot")
    J_RL_fun = kindyn_model.kindyn.jacobian_fun("RL_foot")
    J_RR_fun = kindyn_model.kindyn.jacobian_fun("RR_foot")

    # Symbolic inverse dynamics
    M_sym = mass_matrix_fun(H, joint_pos_sym)
    C_sym = bias_force_fun(H, joint_pos_sym, base_vel_sym, joint_vel_sym)
    g_sym = gravity_fun(H, joint_pos_sym)

    F_FL, F_FR, F_RL, F_RR = cs.vertsplit(f_ext_sym, 3)
    J_FL, J_FR, J_RL, J_RR = (
        f(H, joint_pos_sym)[0:3, :] for f in [J_FL_fun, J_FR_fun, J_RL_fun, J_RR_fun]
    )
    wrench_ext = J_FL.T @ F_FL + J_FR.T @ F_FR + J_RL.T @ F_RL + J_RR.T @ F_RR

    ddq_full_sym = cs.vertcat(base_acc_sym, joint_acc_sym)

    ## FIX: The correct inverse dynamics equation
    tau_full_sym = M_sym @ ddq_full_sym + C_sym + g_sym - wrench_ext

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

    # --- Evaluation Loop ---
    for i in range(num_steps):
        grfs_vec = grf_traj[i, :].copy()
        grfs_vec[0:3] *= contact_sequence[0, i]
        grfs_vec[3:6] *= contact_sequence[1, i]
        grfs_vec[6:9] *= contact_sequence[2, i]
        grfs_vec[9:12] *= contact_sequence[3, i]

        tau_full = inverse_dynamics_fun(
            q_traj[i, 0:3],
            q_traj[i, 3:7],
            q_traj[i, 7:19],
            dq_traj[i, 0:6],
            dq_traj[i, 6:18],
            ddq_traj[i, 0:6],
            ddq_traj[i, 6:18],
            grfs_vec,
        )

        joint_torques_traj[i, :] = tau_full.full().flatten()[6:]

    return joint_torques_traj
