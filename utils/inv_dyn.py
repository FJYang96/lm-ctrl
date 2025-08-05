import casadi as cs
import numpy as np
from liecasadi import SO3


def compute_joint_torques(
    kindyn_model,
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
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
        grf_traj (np.ndarray): The GRF trajectory from the optimizer.
            Shape: (num_steps, 12).
        contact_sequence (np.ndarray): Array of contact flags (1 for stance, 0 for swing).
            Shape: (4, num_steps).
        dt (float): The time step duration.

    Returns:
        np.ndarray: The computed joint torque trajectory.
        Shape: (num_steps, 12).
    """
    # FIX 1: Correctly define num_steps based on the number of trajectory intervals.
    num_steps = grf_traj.shape[0]
    num_joints = 12  # 4 legs * 3 joints

    # FIX 2: Initialize joint_torques_traj with the correct dimensions.
    joint_torques_traj = np.zeros((num_steps, num_joints))

    # We need accelerations, so we'll approximate them via finite differences
    # on the state trajectory. This is a critical step for inverse dynamics.

    # Extract positions and velocities from the state trajectory
    # The state vector is assumed to be:
    # [CoM pos(3), CoM vel(3), RPY(3), angular vel(3), Joint pos(12), Integrals(6)]
    # Total state dimension: 3+3+3+3+12+6 = 30

    # We need positions and velocities for the KinDynComputations
    pos_kindyn_traj = state_traj[:, [0, 1, 2, 6, 7, 8]]  # Base position and RPY
    pos_kindyn_traj = np.hstack(
        [pos_kindyn_traj, state_traj[:, 12:24]]
    )  # Add joint positions

    # We need velocities and accelerations. Let's compute them via finite difference
    dq_traj = np.zeros_like(pos_kindyn_traj)
    ddq_traj = np.zeros_like(pos_kindyn_traj)

    # FIX 3: Correct the loop range to compute velocities for all time steps.
    # state_traj has num_steps + 1 points, allowing for num_steps velocity calculations.
    for i in range(num_steps):
        dq_traj[i, :] = (pos_kindyn_traj[i + 1, :] - pos_kindyn_traj[i, :]) / dt

    # This loop correctly computes num_steps - 1 accelerations.
    for i in range(num_steps - 1):
        ddq_traj[i, :] = (dq_traj[i + 1, :] - dq_traj[i, :]) / dt

    # The KinDynComputations object needs symbolic variables for the function
    # calls. We will create these symbolic variables once.
    # The state is: [base_pos(3), base_rpy(3), joint_pos(12)] = 18 total
    base_pos_sym = cs.SX.sym("base_pos", 3)
    base_rpy_sym = cs.SX.sym("base_rpy", 3)
    joint_pos_sym = cs.SX.sym("joint_pos", 12)
    base_vel_sym = cs.SX.sym("base_vel", 6)  # linear + angular velocities
    joint_vel_sym = cs.SX.sym("joint_vel", 12)
    base_acc_sym = cs.SX.sym("base_acc", 6)  # linear + angular accelerations
    joint_acc_sym = cs.SX.sym("joint_acc", 12)
    f_ext_sym = cs.SX.sym("f_ext", 12)

    # Extract roll, pitch, yaw from base_rpy_sym
    roll, pitch, yaw = base_rpy_sym[0], base_rpy_sym[1], base_rpy_sym[2]

    # Construct homogeneous transformation matrix H
    w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    b_R_w = w_R_b.T
    H = cs.SX.eye(4)
    H[0:3, 0:3] = b_R_w.T
    H[0:3, 3] = base_pos_sym

    mass_matrix_fun = kindyn_model.kindyn.mass_matrix_fun()
    bias_force_fun = kindyn_model.kindyn.bias_force_fun()

    # Get symbolic Jacobians for each foot
    J_FL_fun = kindyn_model.kindyn.jacobian_fun("FL_foot")
    J_FR_fun = kindyn_model.kindyn.jacobian_fun("FR_foot")
    J_RL_fun = kindyn_model.kindyn.jacobian_fun("RL_foot")
    J_RR_fun = kindyn_model.kindyn.jacobian_fun("RR_foot")

    # Create the symbolic function for inverse dynamics
    M_sym = mass_matrix_fun(H, joint_pos_sym)
    C_sym = bias_force_fun(H, joint_pos_sym, base_vel_sym, joint_vel_sym)

    F_FL_sym = f_ext_sym[0:3]
    F_FR_sym = f_ext_sym[3:6]
    F_RL_sym = f_ext_sym[6:9]
    F_RR_sym = f_ext_sym[9:12]

    J_FL_sym = J_FL_fun(H, joint_pos_sym)[0:3, :]
    J_FR_sym = J_FR_fun(H, joint_pos_sym)[0:3, :]
    J_RL_sym = J_RL_fun(H, joint_pos_sym)[0:3, :]
    J_RR_sym = J_RR_fun(H, joint_pos_sym)[0:3, :]

    wrench_ext = (
        J_FL_sym.T @ F_FL_sym
        + J_FR_sym.T @ F_FR_sym
        + J_RL_sym.T @ F_RL_sym
        + J_RR_sym.T @ F_RR_sym
    )

    # Combine accelerations: [base_acc(6), joint_acc(12)]
    ddq_full_sym = cs.vertcat(base_acc_sym, joint_acc_sym)

    tau_full_sym = M_sym @ ddq_full_sym + C_sym - wrench_ext

    inverse_dynamics_fun = cs.Function(
        "inverse_dynamics",
        [
            base_pos_sym,
            base_rpy_sym,
            joint_pos_sym,
            base_vel_sym,
            joint_vel_sym,
            base_acc_sym,
            joint_acc_sym,
            f_ext_sym,
        ],
        [tau_full_sym],
    )

    for i in range(num_steps):
        # Extract components from trajectory
        # pos_kindyn_traj[i, :] = [base_pos(3), base_rpy(3), joint_pos(12)]
        base_pos_vec = pos_kindyn_traj[i, 0:3]
        base_rpy_vec = pos_kindyn_traj[i, 3:6]
        joint_pos_vec = pos_kindyn_traj[i, 6:18]

        # dq_traj[i, :] = [base_vel(3), base_rpy_vel(3), joint_vel(12)]
        # Note: For velocities, we need [linear_vel(3), angular_vel(3)] for base
        base_vel_vec = dq_traj[i, 0:6]  # [linear_vel(3), angular_vel(3)]
        joint_vel_vec = dq_traj[i, 6:18]

        # For the last step, acceleration is not available from finite differencing.
        # A common practice is to use the second-to-last acceleration.
        if i < num_steps - 1:
            base_acc_vec = ddq_traj[i, 0:6]
            joint_acc_vec = ddq_traj[i, 6:18]
        else:
            base_acc_vec = ddq_traj[num_steps - 2, 0:6]
            joint_acc_vec = ddq_traj[num_steps - 2, 6:18]

        # FIX 4: Correct indexing for contact_sequence with shape (4, num_steps).
        # Apply contact forces only for legs in stance.
        grfs_vec = grf_traj[i, :] * np.repeat(contact_sequence[:, i], 3)

        # Compute the full wrench (base and joints)
        tau_full = inverse_dynamics_fun(
            base_pos_vec,
            base_rpy_vec,
            joint_pos_vec,
            base_vel_vec,
            joint_vel_vec,
            base_acc_vec,
            joint_acc_vec,
            grfs_vec,
        )

        # We only need the joint torques (the last 12 values)
        joint_torques_traj[i, :] = tau_full.full().flatten()[6:]

    return joint_torques_traj
