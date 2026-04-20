from typing import Any

import casadi as cs
import numpy as np
from liecasadi import SO3

import go2_config

from .dynamics.model import KinoDynamic_Model

INF = 1e6


def friction_cone_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
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
            go2_config.robot_data.grf_limits * contact_flag
            + go2_config.mpc_config.path_constraint_params["SWING_GRF_EPS"]
            * (1 - contact_flag)
        )  # grf <= EPS when not in contact

        # Friction cone constraints
        # Previously was not active for swing feet, but now is to enforce that swing feet should have zero GRF.
        mu_term = go2_config.experiment.mu_ground * f_z

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
    k: int = 0,
    horizon: int = 1,
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
    foot_height_max = go2_config.mpc_config.path_constraint_params[
        "STANCE_HEIGHT_EPS"
    ] * contact_k + INF * (1 - contact_k)
    foot_height_min = np.zeros(4)

    return foot_heights, foot_height_min, foot_height_max


def no_slip_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    com_position = x_k[0:3]
    com_velocity = x_k[3:6]
    roll, pitch, yaw = x_k[6], x_k[7], x_k[8]
    com_angular_velocity = x_k[9:12]
    joint_positions = x_k[12:24]
    qvel = cs.vertcat(
        com_velocity,
        com_angular_velocity,
        u_k[0:3],
        u_k[3:6],
        u_k[6:9],
        u_k[9:12],
    )
    w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    H = cs.MX.eye(4)
    H[0:3, 0:3] = w_R_b
    H[0:3, 3] = com_position
    v = [
        kindyn_model.jacobian_FL_fun(H, joint_positions)[0:3, :] @ qvel,
        kindyn_model.jacobian_FR_fun(H, joint_positions)[0:3, :] @ qvel,
        kindyn_model.jacobian_RL_fun(H, joint_positions)[0:3, :] @ qvel,
        kindyn_model.jacobian_RR_fun(H, joint_positions)[0:3, :] @ qvel,
    ]
    e = float(go2_config.mpc_config.path_constraint_params["NO_SLIP_EPS"])
    # sumsqr(v) <= e^2  <=>  ||v|| <= e; avoids NaNs from d(||v||)/dv at v=0
    v_sq = cs.vertcat(*[cs.sumsqr(v[i]) for i in range(4)])
    ub = e * e * contact_k + INF * (1 - contact_k)
    return v_sq, np.zeros(4), ub


def joint_limits_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    # Add joint limits to prevent broken configurations
    joint_positions = x_k[12:24]  # 12 joint angles
    return (
        joint_positions,
        go2_config.robot_data.joint_limits_lower,
        go2_config.robot_data.joint_limits_upper,
    )


def input_limits_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    jlim = np.asarray(go2_config.robot_data.joint_velocity_limits, dtype=float).ravel()
    if jlim.size == 1:
        jlim = np.full(12, float(jlim[0]))
    grf_lim = float(np.asarray(go2_config.robot_data.grf_limits).ravel()[0])
    sg_eps = float(
        go2_config.mpc_config.path_constraint_params.get("SWING_GRF_EPS", 0.0)
    )
    lb_f, ub_f = [], []
    for i in range(4):
        c = contact_k[i]
        lo, hi = -grf_lim * c - sg_eps * (1 - c), grf_lim * c + sg_eps * (1 - c)
        lb_f.extend([lo, lo, lo])
        ub_f.extend([hi, hi, hi])
    lb = cs.vertcat(cs.DM(-jlim), cs.vertcat(*lb_f))
    ub = cs.vertcat(cs.DM(jlim), cs.vertcat(*ub_f))
    return u_k, lb, ub


def body_clearance_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
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
    body_half_length, body_half_width, body_half_height = go2_config.body_half_extents

    tilt_clearance = body_half_length * cs.fabs(
        cs.sin(pitch)
    ) + body_half_width * cs.fabs(cs.sin(roll))
    body_clearance_margin = body_half_height + tilt_clearance
    effective_clearance = com_position_z - body_clearance_margin

    # Minimum clearance from config (e.g., 0.02m above ground)
    min_clearance = go2_config.mpc_config.path_constraint_params.get(
        "BODY_CLEARANCE_MIN", 0.02
    )

    return effective_clearance, min_clearance, INF


def link_clearance_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """Constrain calf and head link heights to stay above ground.

    The body_clearance_constraints only checks an approximate COM box.
    This constraint uses FK to check actual link positions that commonly
    penetrate during flips: 4 calves + 2 head links.
    """
    com_position = x_k[0:3]
    roll, pitch, yaw = x_k[6], x_k[7], x_k[8]
    joint_positions = x_k[12:24]

    w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    H = cs.MX.eye(4)
    H[0:3, 0:3] = w_R_b
    H[0:3, 3] = com_position

    link_names = [
        "FL_calf",
        "FR_calf",
        "RL_calf",
        "RR_calf",
        "Head_lower",
        "Head_upper",
    ]
    heights = []
    for link in link_names:
        fk_fun = getattr(kindyn_model, f"forward_kinematics_{link}_fun")
        heights.append(fk_fun(H, joint_positions)[2, 3])

    link_heights = cs.vertcat(*heights)
    min_clearance = 0.01 * np.ones(len(link_names))  # 1cm above ground
    return link_heights, min_clearance, INF * np.ones(len(link_names))


def torque_feasibility_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
    q_ddot_j: cs.MX | None = None,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """Constrain full inverse dynamics joint torques within actuator limits.

    Computes tau = M_jb·a_base + M_jj·qddot_j + h_j - (J^T·F)_j
    and constrains it within actuator torque limits. This accounts for
    inertial coupling, Coriolis, and gravity — not just the GRF contribution.
    """
    com_position = x_k[0:3]
    linear_vel = x_k[3:6]
    roll, pitch, yaw = x_k[6], x_k[7], x_k[8]
    angular_vel = x_k[9:12]
    joint_positions = x_k[12:24]
    joint_velocities = u_k[0:12]
    forces = u_k[12:24]

    w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    H = cs.MX.eye(4)
    H[0:3, 0:3] = w_R_b
    H[0:3, 3] = com_position

    # Full 18x18 mass matrix
    M = kindyn_model.mass_mass_fun(H, joint_positions)
    M_bb = M[0:6, 0:6]
    M_bj = M[0:6, 6:18]
    M_jb = M[6:18, 0:6]
    M_jj = M[6:18, 6:18]

    # Bias forces (Coriolis + gravity)
    base_vel = cs.vertcat(linear_vel, angular_vel)
    h = kindyn_model.bias_force_fun(H, joint_positions, base_vel, joint_velocities)
    h_b = h[0:6]
    h_j = h[6:18]

    # J^T·F summed across all feet
    jacobian_funs = [
        kindyn_model.jacobian_FL_fun,
        kindyn_model.jacobian_FR_fun,
        kindyn_model.jacobian_RL_fun,
        kindyn_model.jacobian_RR_fun,
    ]
    JtF = cs.MX.zeros(18)
    for leg_idx in range(4):
        f_foot = forces[leg_idx * 3 : leg_idx * 3 + 3]
        contact_flag = contact_k[leg_idx]
        J_full = jacobian_funs[leg_idx](H, joint_positions)[0:3, :]
        JtF += J_full.T @ (f_foot * contact_flag)

    JtF_b = JtF[0:6]
    JtF_j = JtF[6:18]

    if q_ddot_j is None:
        q_ddot_j = cs.MX.zeros(12)

    # Base acceleration (same equation as forward_dynamics):
    # M_bb · a_base = -h_b + J^T·F_b - M_bj · qddot_j
    a_base = cs.inv(M_bb) @ (-h_b + JtF_b - M_bj @ q_ddot_j)

    # Full inverse dynamics joint torque:
    # tau = M_jb · a_base + M_jj · qddot_j + h_j - (J^T·F)_j
    tau_joints = M_jb @ a_base + M_jj @ q_ddot_j + h_j - JtF_j

    # 80% of actual limits to leave headroom for PD tracking corrections
    torque_limits = go2_config.robot_data.joint_efforts * 0.8

    return tau_joints, -torque_limits, torque_limits


def angular_momentum_flight_constraint(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
    x_prev: cs.MX | None = None,
    u_prev: cs.MX | None = None,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """Conserve centroidal angular momentum during flight.

    During flight (no ground contact), angular momentum about the COM is
    conserved. The MPC's Euler integration doesn't enforce this exactly,
    so the solver can exploit numerical drift to gain free rotation.
    This constraint closes that loophole.

    Uses ADAM's centroidal momentum matrix: h = A_G(q) · v_gen,
    where h[3:6] is the angular momentum about the COM.
    """
    if x_prev is None or u_prev is None:
        # No previous timestep — can't compute change, skip
        return cs.MX.zeros(3), -INF * np.ones(3), INF * np.ones(3)

    def _centroidal_angular_momentum(x: cs.MX, u: cs.MX) -> cs.MX:
        com_pos = x[0:3]
        lin_vel = x[3:6]
        roll, pitch, yaw = x[6], x[7], x[8]
        ang_vel = x[9:12]
        joint_pos = x[12:24]
        joint_vel = u[0:12]

        w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
        H = cs.MX.eye(4)
        H[0:3, 0:3] = w_R_b
        H[0:3, 3] = com_pos

        # Centroidal momentum matrix A_G: (6, 18)
        # h = A_G · [v_base(6); q̇(12)]  →  h[3:6] = angular momentum about COM
        A_G = kindyn_model.centroidal_momentum_matrix_fun(H, joint_pos)
        v_gen = cs.vertcat(lin_vel, ang_vel, joint_vel)
        h = A_G @ v_gen  # (6,)
        return h[3:6]  # angular momentum about COM (3,)

    L_k = _centroidal_angular_momentum(x_k, u_k)
    L_prev = _centroidal_angular_momentum(x_prev, u_prev)

    # Only constrain during flight (all contacts = 0)
    any_contact = contact_k[0] + contact_k[1] + contact_k[2] + contact_k[3]
    is_flight = 1.0 - cs.fmin(any_contact, 1.0)  # 1 if flight, 0 if stance

    delta_L = (L_k - L_prev) * is_flight
    tol = 0.5 * np.ones(3)  # small tolerance for integration discretization
    return delta_L, -tol, tol


def joint_acceleration_constraint(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
    k: int = 0,
    horizon: int = 1,
    u_prev: cs.MX | None = None,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """Bound finite-difference joint acceleration to actuator-feasible range."""
    if u_prev is None:
        return cs.MX.zeros(12), -INF * np.ones(12), INF * np.ones(12)
    dt = float(go2_config.mpc_config.mpc_dt)
    q_ddot = (u_k[0:12] - u_prev[0:12]) / dt
    bound = np.asarray(go2_config.joint_acceleration_limits, dtype=float)
    return q_ddot, -bound, bound
