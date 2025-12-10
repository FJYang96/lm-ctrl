from typing import Any

import casadi as cs
import numpy as np
from liecasadi import SO3

from .dynamics.model import KinoDynamic_Model

INF = 1e6


def friction_cone_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """Add friction cone constraints exactly like working Acados version."""
    # Key insight: The working version uses contact_sequence to DISABLE constraints
    # When contact_flag = 0, ALL force constraints become [-1e6, 1e6] (no constraint) <== this is not correct
    # When contact_flag = 1, proper friction cone constraints are applied

    # Extract forces for each foot
    forces = u_k[12:24]

    expr_list, min_list, max_list = [], [], []

    for foot_idx in range(4):
        f_x = forces[foot_idx * 3]
        f_y = forces[foot_idx * 3 + 1]
        f_z = forces[foot_idx * 3 + 2]
        contact_flag = contact_k[foot_idx]

        # Normal force constraints
        expr_list.append(f_z)
        min_list.append(0)
        max_list.append(
            config.robot_data.grf_limits * contact_flag
            + config.mpc_config.path_constraint_params["SWING_GRF_EPS"]
            * (1 - contact_flag)
        )  # grf <= EPS when not in contact

        # Friction cone constraints
        # Previously was not active for swing feet, but now is to enforce that swing feet should have zero GRF.
        mu_term = config.experiment.mu_ground * f_z

        expr_list.append(f_x)
        min_list.append(-mu_term)
        max_list.append(mu_term)

        expr_list.append(f_y)
        min_list.append(-mu_term)
        max_list.append(mu_term)

    return cs.vertcat(*expr_list), cs.vertcat(*min_list), cs.vertcat(*max_list)


def foot_height_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """
    Add foot height constraints based on the contact schedule.
    - Stance feet: Constrain vertical position to be slightly above zero.
    - Swing feet: Constrain vertical position to be non-negative (above ground).
    """
    # ... (the forward kinematics part remains the same) ...
    com_position = x_k[0:3]
    roll = x_k[6]
    pitch = x_k[7]
    yaw = x_k[8]
    joint_positions = x_k[12:24]

    w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    H = cs.MX.eye(4)
    H[0:3, 0:3] = w_R_b
    H[0:3, 3] = com_position

    foot_height_fl = kindyn_model.forward_kinematics_FL_fun(H, joint_positions)[2, 3]
    foot_height_fr = kindyn_model.forward_kinematics_FR_fun(H, joint_positions)[2, 3]
    foot_height_rl = kindyn_model.forward_kinematics_RL_fun(H, joint_positions)[2, 3]
    foot_height_rr = kindyn_model.forward_kinematics_RR_fun(H, joint_positions)[2, 3]

    foot_heights = cs.vertcat(
        foot_height_fl, foot_height_fr, foot_height_rl, foot_height_rr
    )
    foot_height_max = config.mpc_config.path_constraint_params[
        "STANCE_HEIGHT_EPS"
    ] * contact_k + INF * (1 - contact_k)
    foot_height_min = np.zeros(4)

    return foot_heights, foot_height_min, foot_height_max


def foot_velocity_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    com_position = x_k[0:3]
    com_velocity = x_k[3:6]
    roll = x_k[6]
    pitch = x_k[7]
    yaw = x_k[8]
    com_angular_velocity = x_k[9:12]
    joint_positions = x_k[12:24]

    qvel_joints_FL = u_k[0:3]
    qvel_joints_FR = u_k[3:6]
    qvel_joints_RL = u_k[6:9]
    qvel_joints_RR = u_k[9:12]

    # Create homogeneous transformation matrix
    w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    H = cs.MX.eye(4)
    H[0:3, 0:3] = w_R_b
    H[0:3, 3] = com_position

    qvel = cs.vertcat(
        com_velocity,
        com_angular_velocity,
        qvel_joints_FL,
        qvel_joints_FR,
        qvel_joints_RL,
        qvel_joints_RR,
    )

    foot_vel_FL = kindyn_model.jacobian_FL_fun(H, joint_positions)[0:3, :] @ qvel
    foot_vel_FR = kindyn_model.jacobian_FR_fun(H, joint_positions)[0:3, :] @ qvel
    foot_vel_RL = kindyn_model.jacobian_RL_fun(H, joint_positions)[0:3, :] @ qvel
    foot_vel_RR = kindyn_model.jacobian_RR_fun(H, joint_positions)[0:3, :] @ qvel

    # Stack only x and y foot velocities (exclude z-direction)
    # The z-direction no-slip constraint is redundant because foot-height constraints already enforce it
    foot_velocities = cs.vertcat(
        foot_vel_FL[0:2], foot_vel_FR[0:2], foot_vel_RL[0:2], foot_vel_RR[0:2]
    )  # Dimension: 4 x 2 = 8 (only x and y components)
    foot_velocity_min = cs.repmat(
        -config.mpc_config.path_constraint_params["NO_SLIP_EPS"] * contact_k
        - INF * (1 - contact_k),
        2,  # Only 2 components (x, y) instead of 3
        1,
    ).reshape((-1, 1))
    foot_velocity_max = cs.repmat(
        config.mpc_config.path_constraint_params["NO_SLIP_EPS"] * contact_k
        + INF * (1 - contact_k),
        2,  # Only 2 components (x, y) instead of 3
        1,
    ).reshape((-1, 1))

    return foot_velocities, foot_velocity_min, foot_velocity_max


def joint_limits_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    # Add joint limits to prevent broken configurations
    joint_positions = x_k[12:24]  # 12 joint angles
    return (
        joint_positions,
        config.robot_data.joint_limits_lower,
        config.robot_data.joint_limits_upper,
    )


def input_limits_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    lb = np.concatenate(
        (
            np.ones(12) * -config.robot_data.joint_velocity_limits,
            np.ones(12) * -config.robot_data.grf_limits,
        )
    )
    ub = np.concatenate(
        (
            np.ones(12) * config.robot_data.joint_velocity_limits,
            np.ones(12) * config.robot_data.grf_limits,
        )
    )
    return u_k, lb, ub


def body_clearance_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """
    Add body clearance constraints to ensure all parts of the robot body remain above ground.

    This constraint considers the COM height and the robot's body dimensions (roll, pitch)
    to ensure that even when the body is tilted, the lowest point of the body remains
    above a minimum clearance height.

    We need to account for:
    - COM position z-coordinate
    - Body roll and pitch that could bring edges of the body closer to ground
    - Half of the body height dimension to get the lowest point
    """
    com_position_z = x_k[2]  # z-coordinate of COM
    roll = x_k[6]
    pitch = x_k[7]

    # Safety margin accounts for body dimensions and tilt
    # For small angles: additional_clearance ≈ body_length/2 * |pitch| + body_width/2 * |roll|
    body_half_length = 0.15
    body_half_width = 0.10
    body_half_height = 0.05

    tilt_clearance = body_half_length * cs.fabs(pitch) + body_half_width * cs.fabs(roll)
    body_clearance_margin = body_half_height + tilt_clearance
    effective_clearance = com_position_z - body_clearance_margin

    # Minimum clearance from config (e.g., 0.02m above ground)
    min_clearance = config.mpc_config.path_constraint_params.get(
        "BODY_CLEARANCE_MIN", 0.02
    )

    return effective_clearance, min_clearance, INF
