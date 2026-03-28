# Adapted from https://github.com/iit-DLSLab/Quadruped-PyMPC
# Full-body 18-DOF kinodynamic model for Go2 quadruped.

from __future__ import annotations

import os
from typing import Any

import casadi as cs
import gym_quadruped
import numpy as np

import go2_config

try:
    from acados_template import AcadosModel
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False
from adam import Representations
from adam.casadi import KinDynComputations
from liecasadi import SO3

gym_quadruped_path = os.path.dirname(gym_quadruped.__file__)


class KinoDynamic_Model:
    """Full 18-DOF kinodynamic model for Go2 quadruped.

    Dynamics: M_bb · a_base + M_bj · q̈_j = -η_base + J^T · F
    Solved as: a_base = M_bb^{-1} · (-η_base + J^T·F - M_bj · q̈_j)
    """

    def __init__(self) -> None:
        self.kindyn = KinDynComputations(
            urdfstring=go2_config.robot_data.urdf_filename,
            joints_name_list=list(go2_config.JOINT_ORDER),
        )
        self.kindyn.set_frame_velocity_representation(
            representation=Representations.MIXED_REPRESENTATION
        )
        self.mass_mass_fun = self.kindyn.mass_matrix_fun()
        self.com_position_fun = self.kindyn.CoM_position_fun()
        self.bias_force_fun = self.kindyn.bias_force_fun()
        self.gravity_fun = self.kindyn.gravity_term_fun()
        self.coriolis_fun = self.kindyn.coriolis_term_fun()

        for foot in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            prefix = foot.split("_")[0]
            setattr(self, f"forward_kinematics_{prefix}_fun",
                    self.kindyn.forward_kinematics_fun(foot))
            setattr(self, f"jacobian_{prefix}_fun",
                    self.kindyn.jacobian_fun(foot))

        # FK for body links that can penetrate ground (calves + head)
        for link in ["FL_calf", "FR_calf", "RL_calf", "RR_calf",
                      "Head_lower", "Head_upper"]:
            setattr(self, f"forward_kinematics_{link}_fun",
                    self.kindyn.forward_kinematics_fun(link))

        self._build_symbolic_variables()

    def _build_symbolic_variables(self) -> None:
        """Build CasADi symbolic state/input/parameter vectors."""
        # State: [com_pos(3), com_vel(3), euler(3), omega(3), joints(12), integrals(6)] = 30
        self.states = cs.vertcat(
            cs.SX.sym("com_pos", 3), cs.SX.sym("com_vel", 3),
            cs.SX.sym("euler", 3), cs.SX.sym("omega", 3),
            cs.SX.sym("joints_FL", 3), cs.SX.sym("joints_FR", 3),
            cs.SX.sym("joints_RL", 3), cs.SX.sym("joints_RR", 3),
            cs.SX.sym("integrals", 6),
        )
        self.states_dot = cs.vertcat(
            cs.SX.sym("com_vel_dot", 3), cs.SX.sym("com_acc", 3),
            cs.SX.sym("euler_rates", 3), cs.SX.sym("angular_acc", 3),
            cs.SX.sym("jvel_FL", 3), cs.SX.sym("jvel_FR", 3),
            cs.SX.sym("jvel_RL", 3), cs.SX.sym("jvel_RR", 3),
            cs.SX.sym("integral_dots", 6),
        )
        # Input: [joint_vel(12), foot_forces(12)] = 24
        self.inputs = cs.vertcat(
            cs.SX.sym("joint_vel", 12), cs.SX.sym("foot_forces", 12),
        )
        self.y_ref = cs.vertcat(self.states, self.inputs)

        # Parameters: stance(4), q_ddot_j(12) = 16
        self.stance_param = cs.SX.sym("stance", 4)
        self.q_ddot_j = cs.SX.sym("q_ddot_j", 12)

    def forward_dynamics(self, states: Any, inputs: Any, param: Any) -> cs.SX:
        """Compute full 18-DOF forward dynamics.

        M_bb · a_base = -η_base + J^T·F - M_bj · q̈_j
        """
        foot_forces = [inputs[12 + i * 3:12 + (i + 1) * 3] for i in range(4)]
        stance = [param[i:i + 1] for i in range(4)]
        q_ddot_j = param[4:16]

        com_pos = states[0:3]
        joint_pos = states[12:24]
        linear_com_vel = states[3:6]
        w = states[9:12]
        roll, pitch, yaw = states[6], states[7], states[8]

        # Euler angle rates
        T = cs.SX.eye(3)
        T[1, 1] = cs.cos(roll)
        T[2, 2] = cs.cos(pitch) * cs.cos(roll)
        T[2, 1] = -cs.sin(roll)
        T[0, 2] = -cs.sin(pitch)
        T[1, 2] = cs.cos(pitch) * cs.sin(roll)
        euler_rates = cs.inv(T) @ w

        # Homogeneous transform
        w_R_b = SO3.from_euler(np.array([roll, pitch, yaw])).as_matrix()
        H = cs.SX.eye(4)
        H[0:3, 0:3] = w_R_b
        H[0:3, 3] = com_pos

        # Foot positions (stored for constraint access)
        fk_funs = [self.forward_kinematics_FL_fun, self.forward_kinematics_FR_fun,
                   self.forward_kinematics_RL_fun, self.forward_kinematics_RR_fun]
        jac_funs = [self.jacobian_FL_fun, self.jacobian_FR_fun,
                    self.jacobian_RL_fun, self.jacobian_RR_fun]
        for fk, name in zip(fk_funs, ["fl", "fr", "rl", "rr"]):
            setattr(self, f"foot_position_{name}", fk(H, joint_pos)[0:3, 3])

        # J^T · F summed across all feet (base DOFs only)
        u_wrenches = sum(
            jac_funs[i](H, joint_pos)[0:3, :].T @ foot_forces[i] @ stance[i]
            for i in range(4)
        )[0:6]

        # Full 18x18 mass matrix — extract M_bb (6x6) and M_bj (6x12)
        M_full = self.mass_mass_fun(H, joint_pos)
        M_bb = M_full[0:6, 0:6]
        M_bj = M_full[0:6, 6:18]

        # Bias forces (Coriolis + gravity)
        base_vel = cs.vertcat(linear_com_vel, w)
        eta = self.bias_force_fun(H, joint_pos, base_vel, inputs[0:12])[0:6]

        # Full Newton-Euler: M_bb · a_base = -η + J^T·F - M_bj · q̈_j
        acc = cs.inv(M_bb) @ (-eta + u_wrenches - M_bj @ q_ddot_j)

        # Integral states
        integrals = states[24:]
        integrals[0] += states[2]
        integrals[1:4] += states[3:6]
        integrals[4] += roll
        integrals[5] += pitch

        return cs.vertcat(
            linear_com_vel, acc[0:3], euler_rates, acc[3:6],
            inputs[0:3], inputs[3:6], inputs[6:9], inputs[9:12],
            integrals,
        )

    def export_robot_model(self) -> Any:
        """Export the dynamics model for the MPC solver."""
        self.param = cs.vertcat(self.stance_param, self.q_ddot_j)
        f_expl = self.forward_dynamics(self.states, self.inputs, self.param)

        if not ACADOS_AVAILABLE:
            class MinimalModel:
                def __init__(self, x: cs.MX, u: cs.MX, p: cs.MX, f: cs.MX) -> None:
                    self.x, self.u, self.p = x, u, p
                    self.f_expl_expr = f
                    self.name = "kinodynamic_model"
            return MinimalModel(self.states, self.inputs, self.param, f_expl)

        acados_model = AcadosModel()
        acados_model.f_impl_expr = self.states_dot - f_expl
        acados_model.f_expl_expr = f_expl
        acados_model.x = self.states
        acados_model.xdot = self.states_dot
        acados_model.u = self.inputs
        acados_model.p = self.param
        acados_model.name = "kinodynamic_model"
        return acados_model
