"""Feedforward torque computation via full inverse dynamics.

Computes tau_ff = M_jb·a_base + M_jj·q̈_j + h_j − (J^T·F)_j
using the same equation as MPC's torque_feasibility_constraints.
This accounts for inertial coupling, Coriolis, and gravity —
not just the GRF contribution.

Used as a feedforward term in the PD controller:
  torque = Kp*(target-actual) + Kd*(vel_err) + tau_ff
"""

from __future__ import annotations

import casadi as cs
import numpy as np
from liecasadi import SO3

from mpc.dynamics.model import KinoDynamic_Model
from utils.conversion import (
    MPC_X_BASE_ANG,
    MPC_X_BASE_EUL,
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_Q_JOINTS,
)

from .reference import ReferenceTrajectory


class FeedforwardComputer:
    """Computes full inverse dynamics feedforward torques."""

    def __init__(self, kindyn_model: KinoDynamic_Model):
        """Build CasADi function for full ID: tau = M_jb·a_base + M_jj·q̈_j + h_j − (J^T·F)_j."""
        base_pos_sym = cs.SX.sym("base_pos", 3)
        base_rpy_sym = cs.SX.sym("base_rpy", 3)
        base_lin_vel_sym = cs.SX.sym("base_lin_vel", 3)
        base_ang_vel_sym = cs.SX.sym("base_ang_vel", 3)
        joint_pos_sym = cs.SX.sym("joint_pos", 12)
        joint_vel_sym = cs.SX.sym("joint_vel", 12)
        grf_sym = cs.SX.sym("grf", 12)
        q_ddot_j_sym = cs.SX.sym("q_ddot_j", 12)

        roll, pitch, yaw = base_rpy_sym[0], base_rpy_sym[1], base_rpy_sym[2]
        w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
        H = cs.SX.eye(4)
        H[0:3, 0:3] = w_R_b
        H[0:3, 3] = base_pos_sym

        # Full 18x18 mass matrix
        M = kindyn_model.kindyn.mass_matrix_fun()(H, joint_pos_sym)
        M_bb = M[0:6, 0:6]
        M_bj = M[0:6, 6:18]
        M_jb = M[6:18, 0:6]
        M_jj = M[6:18, 6:18]

        # Bias forces (Coriolis + gravity)
        base_vel = cs.vertcat(base_lin_vel_sym, base_ang_vel_sym)
        h = kindyn_model.kindyn.bias_force_fun()(H, joint_pos_sym, base_vel, joint_vel_sym)
        h_b = h[0:6]
        h_j = h[6:18]

        # J^T · F summed across all feet
        jac_funs = {
            "FL_foot": kindyn_model.kindyn.jacobian_fun("FL_foot"),
            "FR_foot": kindyn_model.kindyn.jacobian_fun("FR_foot"),
            "RL_foot": kindyn_model.kindyn.jacobian_fun("RL_foot"),
            "RR_foot": kindyn_model.kindyn.jacobian_fun("RR_foot"),
        }
        JtF = cs.SX.zeros(18)
        for i, foot in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            f_foot = grf_sym[i * 3 : i * 3 + 3]
            J_lin = jac_funs[foot](H, joint_pos_sym)[0:3, :]
            JtF += J_lin.T @ f_foot

        JtF_b = JtF[0:6]
        JtF_j = JtF[6:18]

        # Base acceleration: M_bb · a_base = -h_b + J^T·F_b - M_bj · q̈_j
        a_base = cs.inv(M_bb) @ (-h_b + JtF_b - M_bj @ q_ddot_j_sym)

        # Full inverse dynamics joint torque:
        # tau = M_jb · a_base + M_jj · q̈_j + h_j - (J^T·F)_j
        tau_joints = M_jb @ a_base + M_jj @ q_ddot_j_sym + h_j - JtF_j

        self._ff_fun = cs.Function(
            "feedforward_torque_full_id",
            [base_pos_sym, base_rpy_sym, base_lin_vel_sym, base_ang_vel_sym,
             joint_pos_sym, joint_vel_sym, grf_sym, q_ddot_j_sym],
            [tau_joints],
        )

    def compute(
        self,
        base_pos: np.ndarray,
        base_rpy: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        grf: np.ndarray,
        q_ddot_j: np.ndarray,
    ) -> np.ndarray:
        """Compute full ID feedforward torques for one timestep. Returns (12,)."""
        return np.array(self._ff_fun(
            base_pos, base_rpy, base_lin_vel, base_ang_vel,
            joint_pos, joint_vel, grf, q_ddot_j,
        )).flatten()

    def precompute_trajectory(self, ref: ReferenceTrajectory) -> np.ndarray:
        """Precompute full ID feedforward torques for entire reference. Returns (N, 12)."""
        N = ref.max_phase
        ff_torques = np.zeros((N, 12))
        for k in range(N):
            # Joint acceleration from finite differences
            if k > 0:
                q_ddot_j = (ref.joint_vel_traj[k] - ref.joint_vel_traj[k - 1]) / ref.control_dt
            else:
                q_ddot_j = np.zeros(12)

            ff_torques[k] = self.compute(
                ref.state_traj[k, MPC_X_BASE_POS],
                ref.state_traj[k, MPC_X_BASE_EUL],
                ref.state_traj[k, MPC_X_BASE_VEL],
                ref.state_traj[k, MPC_X_BASE_ANG],
                ref.state_traj[k, MPC_X_Q_JOINTS],
                ref.joint_vel_traj[k] if k < N else ref.joint_vel_traj[N - 1],
                ref.grf_traj[k],
                q_ddot_j,
            )
        return ff_torques
