"""Feedforward torque computation: tau_ff = J^T · F.

Computes the joint torques implied by reference ground reaction forces
via the leg Jacobian transpose. Used as a feedforward term in the PD
controller: torque = Kp*(target-actual) + Kd*(vel_err) + tau_ff.

Same computation as utils/inv_dyn.py lines 86-112, but standalone.
"""

from __future__ import annotations

import casadi as cs
import numpy as np
from liecasadi import SO3

from mpc.dynamics.model import KinoDynamic_Model
from utils.conversion import MPC_X_BASE_EUL, MPC_X_BASE_POS, MPC_X_Q_JOINTS

from .reference import ReferenceTrajectory


class FeedforwardComputer:
    """Computes tau_ff = J^T · F for all 4 feet, extracting joint torques."""

    def __init__(self, kindyn_model: KinoDynamic_Model):
        """Build CasADi function: (base_pos, base_rpy, joint_pos, grf) -> tau_ff (12D)."""
        base_pos_sym = cs.SX.sym("base_pos", 3)
        base_rpy_sym = cs.SX.sym("base_rpy", 3)
        joint_pos_sym = cs.SX.sym("joint_pos", 12)
        grf_sym = cs.SX.sym("grf", 12)

        # Homogeneous transform from base pose
        roll, pitch, yaw = base_rpy_sym[0], base_rpy_sym[1], base_rpy_sym[2]
        w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
        H = cs.SX.eye(4)
        H[0:3, 0:3] = w_R_b
        H[0:3, 3] = base_pos_sym

        # Linear Jacobians (3x18) for each foot — take rows [0:3] of full (6x18) Jacobian
        J_FL = kindyn_model.kindyn.jacobian_fun("FL_foot")(H, joint_pos_sym)[0:3, :]
        J_FR = kindyn_model.kindyn.jacobian_fun("FR_foot")(H, joint_pos_sym)[0:3, :]
        J_RL = kindyn_model.kindyn.jacobian_fun("RL_foot")(H, joint_pos_sym)[0:3, :]
        J_RR = kindyn_model.kindyn.jacobian_fun("RR_foot")(H, joint_pos_sym)[0:3, :]

        # J^T · F summed across all feet — full wrench is (18,), take [6:] for joint DOFs
        wrench_full = (
            J_FL.T @ grf_sym[0:3]
            + J_FR.T @ grf_sym[3:6]
            + J_RL.T @ grf_sym[6:9]
            + J_RR.T @ grf_sym[9:12]
        )
        tau_ff_joints = wrench_full[6:]  # skip 6 base DOFs -> (12,)

        self._ff_fun = cs.Function(
            "feedforward_torque",
            [base_pos_sym, base_rpy_sym, joint_pos_sym, grf_sym],
            [tau_ff_joints],
        )

    def compute(
        self,
        base_pos: np.ndarray,
        base_rpy: np.ndarray,
        joint_pos: np.ndarray,
        grf: np.ndarray,
    ) -> np.ndarray:
        """Compute feedforward torques for one timestep. Returns (12,)."""
        return np.array(self._ff_fun(base_pos, base_rpy, joint_pos, grf)).flatten()

    def precompute_trajectory(self, ref: ReferenceTrajectory) -> np.ndarray:
        """Precompute feedforward torques for entire reference. Returns (N, 12).

        Evaluated at the reference configuration (not actual robot state),
        matching OPT-Mimic's approach.
        """
        N = ref.max_phase
        ff_torques = np.zeros((N, 12))
        for k in range(N):
            ff_torques[k] = self.compute(
                ref.state_traj[k, MPC_X_BASE_POS],
                ref.state_traj[k, MPC_X_BASE_EUL],
                ref.state_traj[k, MPC_X_Q_JOINTS],
                ref.grf_traj[k],
            )
        return ff_torques
