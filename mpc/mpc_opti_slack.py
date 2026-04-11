"""MPC with Slack Formulation for constraint hardness analysis.

Extends the base MPC with slack variables to make optimization more robust
and measure the "hardness" of each constraint via slack values.
"""

from __future__ import annotations

from typing import Any

import casadi as cs

import go2_config

from .dynamics.model import KinoDynamic_Model

PHYSICS_CONSTRAINT_NAMES: set[str] = {
    "friction_cone_constraints",
    "foot_height_constraints",
    "no_slip_constraints",
    "joint_limits_constraints",
    "input_limits_constraints",
    "body_clearance_constraints",
    "torque_feasibility_constraints",
    "link_clearance_constraints",
    "angular_momentum_flight_constraint",
}

STATE_ONLY_CONSTRAINT_NAMES = {
    "foot_height_constraints",
    "joint_limits_constraints",
    "body_clearance_constraints",
    "link_clearance_constraints",
}


class QuadrupedMPCOptiSlack:
    """MPC with slack variables for constraint analysis.

    Each soft constraint g_l <= g(x,u) <= g_u becomes:
        g_l - s_lower <= g(x,u) <= g_u + s_upper, s >= 0
    With penalty: J += weight * (||s_lower||^2 + ||s_upper||^2)
    """

    PHYSICS_CONSTRAINT_NAMES = PHYSICS_CONSTRAINT_NAMES
    DEFAULT_SLACK_WEIGHTS: dict[str, float] = {}

    def __init__(
        self,
        model: KinoDynamic_Model,
        config: Any = None,
        build: bool = True,
        use_slack: bool = True,
        slack_weights: dict[str, float] | None = None,
    ) -> None:
        self.horizon = int(
            go2_config.mpc_config.duration / go2_config.mpc_config.mpc_dt
        )
        self.kindyn_model = model
        self.use_slack = use_slack
        self.slack_weights = self.DEFAULT_SLACK_WEIGHTS.copy()
        if slack_weights:
            self.slack_weights.update(slack_weights)

        acados_model = self.kindyn_model.export_robot_model()
        self.states_dim = acados_model.x.size()[0]
        self.inputs_dim = acados_model.u.size()[0]

        self.pre_flight_steps = int(
            go2_config.mpc_config.pre_flight_stance_duration
            / go2_config.mpc_config.mpc_dt
        )
        self.flight_steps = int(
            go2_config.mpc_config.flight_duration / go2_config.mpc_config.mpc_dt
        )
        self.landing_start = self.pre_flight_steps + self.flight_steps

        self.opti = cs.Opti()
        self.slack_variables: dict[tuple[str, int], tuple[cs.MX, cs.MX, int]] = {}
        self.slack_penalty_cost = 0
        self._last_solution = None
        self._debug_accessor = None

        # Decision variables
        self.X = self.opti.variable(self.states_dim, self.horizon + 1)
        self.U = self.opti.variable(self.inputs_dim, self.horizon)
        self.P_contact = self.opti.parameter(4, self.horizon)
        self.P_initial_state = self.opti.parameter(self.states_dim)
        self.P_X_ref = self.opti.parameter(self.states_dim, self.horizon + 1)
        self.P_U_ref = self.opti.parameter(self.inputs_dim, self.horizon)

        # Build problem
        self.opti.subject_to(self.X[:, 0] == self.P_initial_state)
        dynamics_fn = cs.Function(
            "dynamics",
            [acados_model.x, acados_model.u, acados_model.p],
            [acados_model.f_expl_expr],
        )
        self._setup_dynamics(dynamics_fn)
        self._setup_path_constraints()
        self._setup_cost()

        solver_opts = dict(go2_config.solver_config)
        self.opti.solver("ipopt", solver_opts)

    def _setup_dynamics(self, dynamics_fn: cs.Function) -> None:
        dt = go2_config.mpc_config.mpc_dt
        for k in range(self.horizon):
            x_k, u_k, x_next = self.X[:, k], self.U[:, k], self.X[:, k + 1]
            contact_k = self.P_contact[:, k]
            # Joint acceleration: q̈_j = (q̇_k - q̇_{k-1}) / dt
            if k >= 1:
                q_ddot_j = (self.U[0:12, k] - self.U[0:12, k - 1]) / dt
            else:
                q_ddot_j = cs.MX.zeros(12)
            param_k = cs.vertcat(contact_k, q_ddot_j)
            self.opti.subject_to(x_next == x_k + dt * dynamics_fn(x_k, u_k, param_k))

    def _add_constraint_with_slack(
        self,
        name: str,
        k: int,
        expr: cs.MX,
        lb: cs.MX,
        ub: cs.MX,
    ) -> None:
        if not self.use_slack or name in self.PHYSICS_CONSTRAINT_NAMES:
            self.opti.subject_to(expr >= lb)
            self.opti.subject_to(expr <= ub)
            return

        n = expr.shape[0] if hasattr(expr, "shape") and len(expr.shape) > 0 else 1
        s_lower = self.opti.variable(n)
        s_upper = self.opti.variable(n)
        self.opti.subject_to(s_lower >= 0)
        self.opti.subject_to(s_upper >= 0)
        self.opti.subject_to(expr >= lb - s_lower)
        self.opti.subject_to(expr <= ub + s_upper)
        self.slack_variables[(name, k)] = (s_lower, s_upper, n)
        weight = self.slack_weights.get(name, 1e3)
        self.slack_penalty_cost += weight * (cs.sumsqr(s_lower) + cs.sumsqr(s_upper))

    def _setup_path_constraints(self) -> None:
        dt = go2_config.mpc_config.mpc_dt
        for k in range(0, self.horizon):
            contact_k = self.P_contact[:, k]
            # Joint acceleration for torque constraint (same as _setup_dynamics)
            q_ddot_j = (self.U[0:12, k] - self.U[0:12, k - 1]) / dt
            for constraint_fn in go2_config.mpc_config.path_constraints:
                name = constraint_fn.__name__
                if name == "torque_feasibility_constraints":
                    expr, lb, ub = constraint_fn(
                        self.X[:, k],
                        self.U[:, k],
                        self.kindyn_model,
                        go2_config,
                        contact_k,
                        k,
                        self.horizon,
                        q_ddot_j=q_ddot_j,
                    )
                elif name == "angular_momentum_flight_constraint":
                    expr, lb, ub = constraint_fn(
                        self.X[:, k],
                        self.U[:, k],
                        self.kindyn_model,
                        go2_config,
                        contact_k,
                        k,
                        self.horizon,
                        x_prev=self.X[:, k - 1],
                        u_prev=self.U[:, k - 1],
                    )
                elif name == "foot_height_constraints" and k == 0:
                    pass
                else:
                    expr, lb, ub = constraint_fn(
                        self.X[:, k],
                        self.U[:, k],
                        self.kindyn_model,
                        go2_config,
                        contact_k,
                        k,
                        self.horizon,
                    )
                self._add_constraint_with_slack(name, k, expr, lb, ub)

        # Terminal state-only constraints
        u_zero = cs.MX.zeros(self.inputs_dim)
        contact_terminal = self.P_contact[:, self.horizon - 1]
        for constraint_fn in go2_config.mpc_config.path_constraints:
            if constraint_fn.__name__ not in STATE_ONLY_CONSTRAINT_NAMES:
                continue
            expr, lb, ub = constraint_fn(
                self.X[:, self.horizon],
                u_zero,
                self.kindyn_model,
                go2_config,
                contact_terminal,
                self.horizon,
                self.horizon,
            )
            self._add_constraint_with_slack(
                constraint_fn.__name__,
                self.horizon,
                expr,
                lb,
                ub,
            )

    def _setup_cost(self) -> None:
        cfg = go2_config.mpc_config
        cost = 0
        for k in range(self.horizon):
            e_base = self.X[0:12, k] - self.P_X_ref[0:12, k]
            e_joint = self.X[12:24, k] - self.P_X_ref[12:24, k]
            e_vel = self.U[0:12, k] - self.P_U_ref[0:12, k]
            e_force = self.U[12:24, k] - self.P_U_ref[12:24, k]
            cost += cs.mtimes([e_base.T, cfg.q_base, e_base])
            cost += cs.mtimes([e_joint.T, cfg.q_joint, e_joint])
            cost += cs.mtimes([e_vel.T, cfg.r_joint_vel, e_vel])
            cost += cs.mtimes([e_force.T, cfg.r_forces, e_force])

        e_term_base = self.X[0:12, self.horizon] - self.P_X_ref[0:12, self.horizon]
        e_term_joint = self.X[12:24, self.horizon] - self.P_X_ref[12:24, self.horizon]
        cost += 2.0 * cs.mtimes([e_term_base.T, cfg.q_terminal_base, e_term_base])
        cost += 2.0 * cs.mtimes([e_term_joint.T, cfg.q_terminal_joint, e_term_joint])

        if self.use_slack:
            cost += self.slack_penalty_cost
        self.opti.minimize(cost)
