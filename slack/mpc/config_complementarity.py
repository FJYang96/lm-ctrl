"""
Complementarity-based contact optimization configuration.

Key concepts:
1. Contact forces and velocities must satisfy complementarity: f_i * v_i = 0
2. We relax this using a barrier function: f_i * v_i <= ε
3. Contact forces are only non-zero when in contact (determined by gait schedule)
4. Foot velocities should be zero during stance phase
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import casadi as cs
import numpy as np

from .constraints import (
    body_clearance_constraints,
    foot_height_constraints,
    friction_cone_constraints,
    input_limits_constraints,
    joint_limits_constraints,
)
from .dynamics.model import KinoDynamic_Model
from .mpc_config import MPCConfig


def complementarity_constraints(
    x_k: cs.MX,
    u_k: cs.MX,
    kindyn_model: KinoDynamic_Model,
    config: Any,
    contact_k: cs.MX,
) -> tuple[cs.MX, cs.MX, cs.MX]:
    """
    Implement relaxed complementarity constraints: f_normal * v_normal <= epsilon

    This constraint ensures that contact forces and velocities don't both be
    significantly non-zero, which would violate physical contact mechanics.

    For each foot:
    - f_z: normal force (from u_k[12:24])
    - v_z: normal velocity (from foot Jacobian * qvel)
    - Constraint: f_z * v_z <= epsilon

    The constraint is only active during stance phase (contact_k = 1).
    """
    # Extract state components
    com_position = x_k[0:3]
    com_velocity = x_k[3:6]
    roll = x_k[6]
    pitch = x_k[7]
    yaw = x_k[8]
    com_angular_velocity = x_k[9:12]
    joint_positions = x_k[12:24]

    # Extract joint velocities and forces
    qvel_joints_FL = u_k[0:3]
    qvel_joints_FR = u_k[3:6]
    qvel_joints_RL = u_k[6:9]
    qvel_joints_RR = u_k[9:12]
    forces = u_k[12:24]  # [FL_xyz, FR_xyz, RL_xyz, RR_xyz]

    # Create homogeneous transformation matrix
    from liecasadi import SO3

    w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
    H = cs.MX.eye(4)
    H[0:3, 0:3] = w_R_b
    H[0:3, 3] = com_position

    # Full velocity vector for Jacobian multiplication
    qvel = cs.vertcat(
        com_velocity,
        com_angular_velocity,
        qvel_joints_FL,
        qvel_joints_FR,
        qvel_joints_RL,
        qvel_joints_RR,
    )

    # Compute foot velocities using Jacobians
    foot_vel_FL = kindyn_model.jacobian_FL_fun(H, joint_positions)[0:3, :] @ qvel
    foot_vel_FR = kindyn_model.jacobian_FR_fun(H, joint_positions)[0:3, :] @ qvel
    foot_vel_RL = kindyn_model.jacobian_RL_fun(H, joint_positions)[0:3, :] @ qvel
    foot_vel_RR = kindyn_model.jacobian_RR_fun(H, joint_positions)[0:3, :] @ qvel

    # Extract normal (z) components
    f_z = cs.vertcat(forces[2], forces[5], forces[8], forces[11])  # Normal forces
    v_z = cs.vertcat(
        foot_vel_FL[2], foot_vel_FR[2], foot_vel_RL[2], foot_vel_RR[2]
    )  # Normal velocities

    # Complementarity products: f_z * v_z
    # We want these to be small (close to zero) during contact
    comp_products = f_z * v_z

    # The constraint is: comp_products <= epsilon (only during stance)
    # During swing, we don't enforce this constraint
    epsilon = config.mpc_config.path_constraint_params.get("COMPLEMENTARITY_EPS", 1e-3)

    # Apply constraint only during stance
    # During swing (contact_k = 0), we relax the constraint to a large value
    expr_list = []
    min_list = []
    max_list = []

    for foot_idx in range(4):
        comp_prod = comp_products[foot_idx]
        contact_flag = contact_k[foot_idx]

        # Constraint: f_z * v_z <= epsilon during stance
        # During swing, we allow up to a large value (effectively no constraint)
        expr_list.append(comp_prod)
        min_list.append(-1e6)  # No lower bound
        max_list.append(epsilon * contact_flag + 1e6 * (1 - contact_flag))

    return cs.vertcat(*expr_list), cs.vertcat(*min_list), cs.vertcat(*max_list)


@dataclass
class ComplementarityMPCConfig(MPCConfig):
    """
    MPC configuration with complementarity constraints for contact handling.

    This extends the base MPCConfig to include complementarity constraints
    that better model contact dynamics by ensuring forces and velocities
    satisfy physical contact mechanics.
    """

    pre_flight_stance_duration: float = 0.3
    flight_duration: float = 0.4
    path_constraints: list[
        Callable[
            [cs.MX, cs.MX, KinoDynamic_Model, Any, cs.MX], tuple[cs.MX, cs.MX, cs.MX]
        ]
    ] = field(
        default_factory=lambda: [
            friction_cone_constraints,
            foot_height_constraints,
            complementarity_constraints,  # Add complementarity
            joint_limits_constraints,
            input_limits_constraints,
            body_clearance_constraints,  # Ensure body stays above ground
        ]
    )
    path_constraint_params: dict[str, float] = field(
        default_factory=lambda: {
            "COMPLEMENTARITY_EPS": 1e-3,
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.02,
            "NO_SLIP_EPS": 0.005,
        }
    )

    @property
    def contact_sequence(self) -> np.ndarray:
        """Generate contact sequence based on gait parameters."""
        if self._contact_sequence is None:
            mpc_horizon = int(self.duration / self.mpc_dt)
            pre_flight_steps = int(self.pre_flight_stance_duration / self.mpc_dt)
            flight_steps = int(self.flight_duration / self.mpc_dt)
            contact_sequence = np.ones((4, mpc_horizon))
            contact_sequence[:, pre_flight_steps : pre_flight_steps + flight_steps] = (
                0.0
            )
            self._contact_sequence = contact_sequence
        return self._contact_sequence
