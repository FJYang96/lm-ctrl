"""
MPC with Slack Formulation for constraint hardness analysis.

This module extends the base MPC with slack variables to:
1. Make the optimization more robust (avoid infeasibility)
2. Measure the "hardness" of each constraint by examining slack values
"""

from __future__ import annotations

import logging
from typing import Any

import casadi as cs
import numpy as np

import go2_config

from .dynamics.model import KinoDynamic_Model

logger = logging.getLogger("llm_integration")

# Module-level alias so that ``from mpc.mpc_opti_slack import
# PHYSICS_CONSTRAINT_NAMES`` works (the canonical set lives on the class).
PHYSICS_CONSTRAINT_NAMES: set[str] = {
    "friction_cone_constraints",
    "foot_height_constraints",
    "foot_velocity_constraints",
    "joint_limits_constraints",
    "input_limits_constraints",
    "body_clearance_constraints",
    "complementarity_constraints",
    "torque_feasibility_constraints",
}


class QuadrupedMPCOptiSlack:
    """
    MPC with slack variables for constraint analysis.

    Each constraint g_l <= g(x,u) <= g_u becomes:
        g_l - s_lower <= g(x,u) <= g_u + s_upper
        s_lower >= 0, s_upper >= 0

    With penalty in cost: J += weight * (||s_lower||² + ||s_upper||²)
    """

    # Physics constraints that must always be hard (no slack).
    # References the module-level set so there is a single source of truth.
    PHYSICS_CONSTRAINT_NAMES = PHYSICS_CONSTRAINT_NAMES

    # Default slack penalty weights for soft (LLM) constraints.
    # LLM constraint names come from the original function (e.g. "height_constraint").
    # If a name is not found here, the fallback weight is 1e3
    # (see _add_constraint_with_slack).
    DEFAULT_SLACK_WEIGHTS: dict[str, float] = {}

    def __init__(
        self,
        model: KinoDynamic_Model,
        config: Any = None,
        build: bool = True,
        use_slack: bool = True,
        slack_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize MPC with optional slack formulation.

        Args:
            model: KinoDynamic_Model instance
            config: Deprecated — ignored. go2_config is used directly.
            build: Whether to build the solver
            use_slack: Whether to use slack formulation
            slack_weights: Custom weights for slack penalties per constraint type
        """
        self.horizon = int(
            go2_config.mpc_config.duration / go2_config.mpc_config.mpc_dt
        )
        self.kindyn_model = model
        self.use_slack = use_slack

        # Merge default weights with custom weights
        self.slack_weights = self.DEFAULT_SLACK_WEIGHTS.copy()
        if slack_weights:
            self.slack_weights.update(slack_weights)

        # Get dimensions from the kinodynamic model
        acados_model = self.kindyn_model.export_robot_model()
        self.states_dim = acados_model.x.size()[0]
        self.inputs_dim = acados_model.u.size()[0]

        # Compute phase boundaries for jumping motion
        self.pre_flight_steps = int(
            go2_config.mpc_config.pre_flight_stance_duration
            / go2_config.mpc_config.mpc_dt
        )
        self.flight_steps = int(
            go2_config.mpc_config.flight_duration / go2_config.mpc_config.mpc_dt
        )
        self.landing_start = self.pre_flight_steps + self.flight_steps

        # Initialize the Opti optimization environment
        self.opti = cs.Opti()

        # Storage for slack variables (for later analysis)
        # Structure: {(constraint_name, timestep_k): (s_lower_var, s_upper_var, n_constraints)}
        self.slack_variables: dict[tuple[str, int], tuple[cs.MX, cs.MX, int]] = {}

        # Store slack penalty cost separately for reporting
        self.slack_penalty_cost = 0

        # Create decision variables
        self._create_decision_variables()

        # Setup the optimization problem structure
        self._setup_optimization_problem()

    def _create_decision_variables(self) -> None:
        """Create the decision variables for the trajectory optimization."""
        # State trajectory: (horizon+1, states_dim)
        self.X = self.opti.variable(self.states_dim, self.horizon + 1)

        # Input trajectory: (horizon, inputs_dim)
        self.U = self.opti.variable(self.inputs_dim, self.horizon)

        # Parameters that can be set at runtime
        self.P_contact = self.opti.parameter(4, self.horizon)  # Contact sequence
        self.P_initial_state = self.opti.parameter(self.states_dim)  # Initial state

        # Per-timestep reference trajectory (cost tracking target + initial guess)
        self.P_X_ref = self.opti.parameter(self.states_dim, self.horizon + 1)
        self.P_U_ref = self.opti.parameter(self.inputs_dim, self.horizon)

        # Robot parameters
        self.P_mu = self.opti.parameter()  # Friction coefficient
        self.P_mass = self.opti.parameter()  # Robot mass
        self.P_inertia = self.opti.parameter(9)  # Flattened inertia matrix

    def _setup_optimization_problem(self) -> None:
        """Setup the complete optimization problem structure."""
        # Set initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.P_initial_state)

        # Setup dynamics constraints
        self._setup_dynamics_constraints()

        # Setup path constraints with slack
        self._setup_path_constraints_with_slack()

        # Setup cost function with slack penalty
        self._setup_cost_function_with_slack()

        # Setup solver options
        self._setup_solver()

    def _setup_dynamics_constraints(self) -> None:
        """Setup dynamics constraints using the kinodynamic model."""
        if not hasattr(self, "_dynamics_fun"):
            self._dynamics_fun = self._create_dynamics_function()

        for k in range(self.horizon):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            x_next = self.X[:, k + 1]

            contact_k = self.P_contact[:, k]
            param_k = cs.vertcat(
                contact_k,
                self.P_mu,
                cs.MX.zeros(4),
                cs.MX.zeros(3),
                cs.MX.zeros(1),
                cs.MX.zeros(6),
                self.P_inertia,
                self.P_mass,
            )

            dt = go2_config.mpc_config.mpc_dt
            f_k = self._dynamics_fun(x_k, u_k, param_k)
            self.opti.subject_to(x_next == x_k + dt * f_k)

    def _create_dynamics_function(self) -> cs.Function:
        """Create a CasADi function for dynamics."""
        acados_model = self.kindyn_model.export_robot_model()
        return cs.Function(
            "dynamics",
            [acados_model.x, acados_model.u, acados_model.p],
            [acados_model.f_expl_expr],
        )

    def _add_constraint_with_slack(
        self,
        constraint_name: str,
        k: int,
        expr: cs.MX,
        lb: cs.MX | np.ndarray,
        ub: cs.MX | np.ndarray,
    ) -> None:
        """
        Add a constraint with optional slack variables.

        Hard constraint: lb <= expr <= ub
        Slack formulation: lb - s_l <= expr <= ub + s_u, s_l >= 0, s_u >= 0
        """
        if not self.use_slack or constraint_name in self.PHYSICS_CONSTRAINT_NAMES:
            # Hard constraints: physics constraints are always hard
            self.opti.subject_to(expr >= lb)
            self.opti.subject_to(expr <= ub)
            return

        # Determine number of constraints
        if hasattr(expr, "shape"):
            n_constraints = expr.shape[0] if len(expr.shape) > 0 else 1
        else:
            n_constraints = 1

        # Create slack variables for this constraint at this timestep
        s_lower = self.opti.variable(n_constraints)
        s_upper = self.opti.variable(n_constraints)

        # Non-negativity constraints on slack variables
        self.opti.subject_to(s_lower >= 0)
        self.opti.subject_to(s_upper >= 0)

        # Relaxed constraints
        self.opti.subject_to(expr >= lb - s_lower)
        self.opti.subject_to(expr <= ub + s_upper)

        # Store slack variables for later analysis
        self.slack_variables[(constraint_name, k)] = (s_lower, s_upper, n_constraints)

        # Add penalty to slack cost
        weight = self.slack_weights.get(constraint_name, 1e3)
        self.slack_penalty_cost += weight * (cs.sumsqr(s_lower) + cs.sumsqr(s_upper))

    # Constraints that only depend on state (x_k), not inputs (u_k).
    # These can be applied at the terminal state where no inputs exist.
    STATE_ONLY_CONSTRAINT_NAMES = {
        "foot_height_constraints",
        "joint_limits_constraints",
        "body_clearance_constraints",
    }

    def _setup_path_constraints_with_slack(self) -> None:
        """Setup path constraints with slack formulation."""
        # Apply all constraints at k=1..horizon-1.
        # k=0 is skipped: the initial state is fixed and the solver needs
        # unconstrained forces at the first timestep to drive dynamics.
        for k in range(1, self.horizon):
            contact_k = self.P_contact[:, k]
            u_k = self.U[:, k]
            x_k = self.X[:, k]

            for constraint_fn in go2_config.mpc_config.path_constraints:
                constraint_name = constraint_fn.__name__

                # Try new signature with k and horizon first, fall back to old signature
                try:
                    expr, lb, ub = constraint_fn(
                        x_k,
                        u_k,
                        self.kindyn_model,
                        go2_config,
                        contact_k,
                        k,
                        self.horizon,
                    )
                except TypeError:
                    # Fall back to old 5-argument signature
                    expr, lb, ub = constraint_fn(
                        x_k, u_k, self.kindyn_model, go2_config, contact_k
                    )

                self._add_constraint_with_slack(constraint_name, k, expr, lb, ub)

        # Apply state-only constraints at the terminal state (k=horizon).
        # No inputs exist at this timestep, so input-dependent constraints are skipped.
        x_terminal = self.X[:, self.horizon]
        # Use last contact column for terminal contact state
        contact_terminal = self.P_contact[:, self.horizon - 1]
        u_zero = cs.MX.zeros(self.inputs_dim)

        for constraint_fn in go2_config.mpc_config.path_constraints:
            constraint_name = constraint_fn.__name__
            if constraint_name not in self.STATE_ONLY_CONSTRAINT_NAMES:
                continue

            try:
                expr, lb, ub = constraint_fn(
                    x_terminal,
                    u_zero,
                    self.kindyn_model,
                    go2_config,
                    contact_terminal,
                    self.horizon,
                    self.horizon,
                )
            except TypeError:
                expr, lb, ub = constraint_fn(
                    x_terminal, u_zero, self.kindyn_model, go2_config, contact_terminal
                )

            self._add_constraint_with_slack(constraint_name, self.horizon, expr, lb, ub)

    def _setup_cost_function_with_slack(self) -> None:
        """Setup the cost function including slack penalties."""
        q_base = go2_config.mpc_config.q_base
        q_joint = go2_config.mpc_config.q_joint
        r_joint_vel = go2_config.mpc_config.r_joint_vel
        r_forces = go2_config.mpc_config.r_forces
        q_terminal_base = go2_config.mpc_config.q_terminal_base
        q_terminal_joint = go2_config.mpc_config.q_terminal_joint

        cost = 0

        # Stage costs: track per-timestep LLM reference trajectory
        for k in range(self.horizon):
            state_error_base = self.X[0:12, k] - self.P_X_ref[0:12, k]
            state_error_joint = self.X[12:24, k] - self.P_X_ref[12:24, k]
            input_error_vel = self.U[0:12, k] - self.P_U_ref[0:12, k]
            input_error_forces = self.U[12:24, k] - self.P_U_ref[12:24, k]

            cost += cs.mtimes([state_error_base.T, q_base, state_error_base])
            cost += cs.mtimes([state_error_joint.T, q_joint, state_error_joint])
            cost += cs.mtimes([input_error_vel.T, r_joint_vel, input_error_vel])
            cost += cs.mtimes([input_error_forces.T, r_forces, input_error_forces])

        # Terminal cost
        terminal_error_base = (
            self.X[0:12, self.horizon] - self.P_X_ref[0:12, self.horizon]
        )
        terminal_error_joint = (
            self.X[12:24, self.horizon] - self.P_X_ref[12:24, self.horizon]
        )
        cost += 2.0 * cs.mtimes(
            [terminal_error_base.T, q_terminal_base, terminal_error_base]
        )
        cost += 2.0 * cs.mtimes(
            [terminal_error_joint.T, q_terminal_joint, terminal_error_joint]
        )

        # Add slack penalty cost
        if self.use_slack:
            cost += self.slack_penalty_cost

        self.opti.minimize(cost)

    def _setup_solver(self) -> None:
        """Setup the solver options."""
        solver_opts = dict(go2_config.solver_config)
        # Cap solve time to prevent excessive runtime with slack variables
        if "ipopt.max_wall_time" not in solver_opts:
            solver_opts["ipopt.max_wall_time"] = 600.0
        self.opti.solver("ipopt", solver_opts)

    def solve_trajectory(
        self,
        initial_state: np.ndarray,
        ref: np.ndarray,
        contact_sequence: np.ndarray,
        ref_trajectory: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Solve the trajectory optimization problem.

        Args:
            initial_state: Initial state vector
            ref: Reference trajectory (shape: states_dim + inputs_dim)
            contact_sequence: Contact sequence array (shape: 4 x horizon)
            ref_trajectory: Optional dict with 'X_ref' (states_dim, horizon+1) and
                'U_ref' (inputs_dim, horizon) used as cost tracking target and
                solver initial guess.
        """
        # Set parameter values
        self.opti.set_value(self.P_initial_state, initial_state)
        self.opti.set_value(self.P_contact, contact_sequence)

        # Build per-timestep reference arrays for the cost function
        if ref_trajectory is not None:
            # Use LLM-generated per-timestep reference as cost target
            X_ref_param = ref_trajectory["X_ref"]
            U_ref_param = ref_trajectory["U_ref"]
        else:
            # Fallback: replicate old phase-based static reference logic
            ref_state = ref[: self.states_dim]
            ref_input = ref[self.states_dim :]

            terminal_state = initial_state.copy()
            terminal_state[0] = ref[0]  # X position from reference
            terminal_state[1] = ref[1]  # Y position from reference
            terminal_state[2] = initial_state[2]  # Z back to initial height
            terminal_state[3:6] = 0.0  # Linear velocity
            terminal_state[9:12] = 0.0  # Angular velocity
            terminal_state[12:24] = initial_state[12:24]  # Joint positions

            X_ref_param = np.zeros((self.states_dim, self.horizon + 1))
            for k in range(self.horizon + 1):
                if k < self.pre_flight_steps:
                    X_ref_param[:, k] = initial_state
                elif k < self.landing_start:
                    X_ref_param[:, k] = ref_state
                else:
                    X_ref_param[:, k] = terminal_state

            U_ref_param = np.tile(ref_input.reshape(-1, 1), (1, self.horizon))

        self.opti.set_value(self.P_X_ref, X_ref_param)
        self.opti.set_value(self.P_U_ref, U_ref_param)

        self.opti.set_value(self.P_mu, go2_config.experiment.mu_ground)
        self.opti.set_value(self.P_mass, go2_config.robot_data.mass)
        self.opti.set_value(self.P_inertia, go2_config.robot_data.inertia.flatten())

        # Determine initial guess: ref_trajectory or heuristic
        if ref_trajectory is not None:
            # Use LLM-generated reference trajectory as initial guess
            logger.info("Using reference trajectory as initial guess")
            X_init = ref_trajectory["X_ref"].copy()
            U_init = ref_trajectory["U_ref"].copy()
        else:
            logger.info("Cold-starting with heuristic initial guess")
            # Initial guess for states (heuristic)
            X_init = np.zeros((self.states_dim, self.horizon + 1))
            for i in range(self.horizon + 1):
                X_init[:, i] = initial_state.copy()

            U_init = np.zeros((self.inputs_dim, self.horizon))
            for i in range(self.horizon):
                U_init[0:12, i] = 0.001 * np.sin(np.arange(12) * 0.1)
                contact_i = (
                    contact_sequence[:, i]
                    if i < contact_sequence.shape[1]
                    else contact_sequence[:, -1]
                )
                for foot in range(4):
                    if contact_i[foot] > 0.5:
                        U_init[12 + foot * 3 + 2, i] = (
                            go2_config.robot_data.mass
                            * go2_config.experiment.gravity_constant
                            / 4
                            * 0.8
                        )

        self.opti.set_initial(self.X, X_init)
        self.opti.set_initial(self.U, U_init)

        # Initialize slack variables to zero
        if self.use_slack:
            for (_, _), (s_l, s_u, _) in self.slack_variables.items():
                self.opti.set_initial(s_l, 0)
                self.opti.set_initial(s_u, 0)

        try:
            sol = self.opti.solve()
            self._last_solution = sol

            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)

            state_traj = X_opt.T
            joint_vel_traj = U_opt[0:12, :].T
            grf_traj = U_opt[12:24, :].T
            status = 0

        except Exception as e:
            print(f"Optimization failed: {e}")
            self._last_solution = None

            # Try to extract debug values even on failure for slack analysis
            try:
                X_opt = self.opti.debug.value(self.X)
                U_opt = self.opti.debug.value(self.U)
                state_traj = X_opt.T
                joint_vel_traj = U_opt[0:12, :].T
                grf_traj = U_opt[12:24, :].T
                self._debug_accessor = self.opti.debug
            except Exception:
                self._debug_accessor = None
                state_traj = np.zeros((self.horizon + 1, self.states_dim))
                joint_vel_traj = np.zeros((self.horizon, 12))
                grf_traj = np.zeros((self.horizon, 12))
            status = 1

        return state_traj, grf_traj, joint_vel_traj, status

    def get_constraint_violations(self) -> dict[str, Any]:
        """
        Analyze the current (possibly infeasible) iterate for debugging.

        Compatible with the base MPC interface so failure feedback works.
        """
        info: dict[str, Any] = {
            "terminal_state": {},
            "trajectory_info": [],
            "summary": [],
        }

        try:
            X_debug = self.opti.debug.value(self.X)

            x_terminal = X_debug[:, -1]
            info["terminal_state"] = {
                "position": {
                    "x": x_terminal[0],
                    "y": x_terminal[1],
                    "z": x_terminal[2],
                },
                "velocity": {
                    "vx": x_terminal[3],
                    "vy": x_terminal[4],
                    "vz": x_terminal[5],
                },
                "orientation": {
                    "roll": x_terminal[6],
                    "pitch": x_terminal[7],
                    "yaw": x_terminal[8],
                },
                "angular_velocity": {
                    "wx": x_terminal[9],
                    "wy": x_terminal[10],
                    "wz": x_terminal[11],
                },
            }

            for k in range(X_debug.shape[1]):
                height = X_debug[2, k]
                if height < 0.0:
                    info["trajectory_info"].append(
                        f"Step {k}: height={height:.3f}m (robot underground!)"
                    )

            if not info["trajectory_info"]:
                info["summary"].append("No physics violations detected")
            else:
                info["summary"].append(
                    f"Physics issues: {len(info['trajectory_info'])}"
                )

        except Exception as e:
            info["summary"].append(f"Could not analyze trajectory: {e}")

        return info

    def analyze_constraint_hardness(self) -> dict[str, dict[str, Any]]:
        """
        Analyze the hardness of each constraint type based on slack values.

        Returns:
            Dictionary mapping constraint names to their hardness metrics
        """
        if not self.use_slack:
            return {}

        # Determine value accessor
        value_fn = None
        if hasattr(self, "_last_solution") and self._last_solution is not None:
            value_fn = self._last_solution.value
        elif hasattr(self, "_debug_accessor") and self._debug_accessor is not None:
            value_fn = self._debug_accessor.value
        else:
            return {}

        results: dict[str, dict[str, Any]] = {}

        for (constraint_name, k), (
            s_l,
            s_u,
            n_constraints,
        ) in self.slack_variables.items():
            try:
                s_l_val = np.array(value_fn(s_l)).flatten()
                s_u_val = np.array(value_fn(s_u)).flatten()
            except Exception:
                continue

            slack_L1 = float(np.sum(np.abs(s_l_val)) + np.sum(np.abs(s_u_val)))
            slack_Linf = float(max(np.max(np.abs(s_l_val)), np.max(np.abs(s_u_val))))

            if constraint_name not in results:
                results[constraint_name] = {
                    "total_slack_L1": 0.0,
                    "max_slack_Linf": 0.0,
                    "total_dims": 0,
                    "active_timesteps": [],
                    "slack_by_timestep": {},
                }

            results[constraint_name]["total_slack_L1"] += slack_L1
            results[constraint_name]["max_slack_Linf"] = max(
                results[constraint_name]["max_slack_Linf"], slack_Linf
            )
            results[constraint_name]["total_dims"] += n_constraints
            results[constraint_name]["slack_by_timestep"][k] = slack_L1

            if slack_L1 > 1e-6:
                results[constraint_name]["active_timesteps"].append(k)

        # Calculate normalized metric
        for name in results:
            total_dims = results[name]["total_dims"]
            if total_dims > 0:
                results[name]["mean_slack_per_dim"] = (
                    results[name]["total_slack_L1"] / total_dims
                )
            else:
                results[name]["mean_slack_per_dim"] = 0.0

        # Sort by max_slack_Linf
        results = dict(sorted(results.items(), key=lambda x: -x[1]["max_slack_Linf"]))

        return results
