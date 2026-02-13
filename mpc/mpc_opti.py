from __future__ import annotations

import logging
from typing import Any

import casadi as cs
import numpy as np

from .dynamics.model import KinoDynamic_Model

logger = logging.getLogger("llm_integration")


def _interpolate_warmstart(
    prev: np.ndarray, target_rows: int, target_cols: int
) -> np.ndarray | None:
    """
    Interpolate a previous solution to match a new horizon size.

    Args:
        prev: Previous solution array (rows x cols)
        target_rows: Expected number of rows (state/input dimension)
        target_cols: Expected number of columns (horizon+1 for X, horizon for U)

    Returns:
        Interpolated array or None if interpolation is not possible.
    """
    if prev is None:
        return None

    # Dimension mismatch in rows (state/input dim changed) → cannot interpolate
    if prev.shape[0] != target_rows:
        return None

    # All-zeros check → debug fallback values were unavailable
    if np.all(prev == 0):
        return None

    prev_cols = prev.shape[1]

    # Same horizon → direct copy
    if prev_cols == target_cols:
        return prev.copy()

    # Different horizon → interpolate along normalized [0,1] time axis per dimension
    old_t = np.linspace(0, 1, prev_cols)
    new_t = np.linspace(0, 1, target_cols)
    result = np.zeros((target_rows, target_cols))
    for i in range(target_rows):
        result[i, :] = np.interp(new_t, old_t, prev[i, :])
    return result


class QuadrupedMPCOpti:
    """
    Quadruped MPC implementation using CasADi's Opti framework.

    This class implements trajectory optimization with universal physics constraints
    (friction cone, no-slip, foot height, dynamics). Task-specific constraints
    including terminal state requirements should be added by the LLM.

    This design allows the LLM to specify appropriate terminal constraints for
    different tasks (e.g., backflip needs terminal pitch ~2π, simple jump needs
    terminal pitch ~0).
    """

    def __init__(
        self, model: KinoDynamic_Model, config: Any, build: bool = True
    ) -> None:
        """
        Initializes the hopping MPC solver using CasADi Opti.

        Args:
            model: KinoDynamic_Model instance
            config: Configuration object with MPC parameters
            build: Whether to build the solver (kept for compatibility)
        """
        self.horizon = int(config.mpc_config.duration / config.mpc_config.mpc_dt)
        self.config = config
        self.kindyn_model = model

        # Get dimensions from the kinodynamic model
        acados_model = self.kindyn_model.export_robot_model()
        self.states_dim = acados_model.x.size()[0]
        self.inputs_dim = acados_model.u.size()[0]

        # Compute phase boundaries for jumping motion
        self.pre_flight_steps = int(
            config.mpc_config.pre_flight_stance_duration / config.mpc_config.mpc_dt
        )
        self.flight_steps = int(
            config.mpc_config.flight_duration / config.mpc_config.mpc_dt
        )
        self.landing_start = self.pre_flight_steps + self.flight_steps

        # Initialize the Opti optimization environment
        self.opti = cs.Opti()

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
        self.P_ref_state = self.opti.parameter(
            self.states_dim
        )  # Reference state (peak)
        self.P_ref_input = self.opti.parameter(self.inputs_dim)  # Reference input
        self.P_initial_state = self.opti.parameter(self.states_dim)  # Initial state
        self.P_terminal_state = self.opti.parameter(
            self.states_dim
        )  # Terminal/landing state

        # Robot parameters
        self.P_mu = self.opti.parameter()  # Friction coefficient
        self.P_grf_min = self.opti.parameter()  # Min ground reaction force
        self.P_grf_max = self.opti.parameter()  # Max ground reaction force
        self.P_mass = self.opti.parameter()  # Robot mass
        self.P_inertia = self.opti.parameter(9)  # Flattened inertia matrix

    def _setup_optimization_problem(self) -> None:
        """Setup the complete optimization problem structure."""
        # Set initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.P_initial_state)

        # Setup cost function
        self._setup_cost_function()

        # Setup dynamics constraints
        self._setup_dynamics_constraints()

        # Setup path constraints
        self._setup_path_constraints()

        # Setup solver options
        self._setup_solver()

    def _setup_dynamics_constraints(self) -> None:
        """Setup dynamics constraints using the kinodynamic model."""
        # Create a CasADi function for dynamics that works with MX
        if not hasattr(self, "_dynamics_fun"):
            self._dynamics_fun = self._create_dynamics_function()

        for k in range(self.horizon):
            # Get current state and input
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            x_next = self.X[:, k + 1]

            # Setup parameters for dynamics
            contact_k = self.P_contact[:, k]
            param_k = cs.vertcat(
                contact_k,  # Contact status
                self.P_mu,  # Friction coefficient
                cs.MX.zeros(4),  # Stance proximity (not used)
                cs.MX.zeros(3),  # Base position (not used)
                cs.MX.zeros(1),  # Base yaw (not used)
                cs.MX.zeros(6),  # External wrench
                self.P_inertia,  # Inertia matrix
                self.P_mass,  # Mass
            )

            # Dynamics constraint: x_{k+1} = x_k + dt * f(x_k, u_k, p_k)
            dt = self.config.mpc_config.mpc_dt
            f_k = self._dynamics_fun(x_k, u_k, param_k)

            # Integration constraint (explicit Euler for now)
            self.opti.subject_to(x_next == x_k + dt * f_k)

    def _create_dynamics_function(self) -> cs.Function:
        """Create a CasADi function for dynamics that works with MX variables."""
        # Get the symbolic expressions from the kinodynamic model
        acados_model = self.kindyn_model.export_robot_model()

        # Create the function directly from the SX expression
        # This automatically handles the conversion to work with different variable types
        return cs.Function(
            "dynamics",
            [acados_model.x, acados_model.u, acados_model.p],
            [acados_model.f_expl_expr],
        )

    def _setup_cost_function(self) -> None:
        """Setup the quadratic tracking cost function with phase-aware references.

        The jumping motion has three phases:
        1. Pre-flight stance: Prepare to jump (reference = initial state)
        2. Flight: Reach peak height (reference = elevated peak state)
        3. Post-landing stance: Land safely (reference = terminal/initial state)
        """
        # Cost weights from config
        q_base = self.config.mpc_config.q_base
        q_joint = self.config.mpc_config.q_joint
        r_joint_vel = self.config.mpc_config.r_joint_vel
        r_forces = self.config.mpc_config.r_forces
        q_terminal_base = self.config.mpc_config.q_terminal_base
        q_terminal_joint = self.config.mpc_config.q_terminal_joint

        # Initialize cost
        cost = 0

        # Stage costs with phase-aware reference selection
        for k in range(self.horizon):
            # Determine which reference to use based on phase
            # Pre-flight: use initial state, Flight: use peak ref, Landing: use terminal
            if k < self.pre_flight_steps:
                # Pre-flight: preparing to jump - reference is initial state
                ref_state = self.P_initial_state
            elif k < self.landing_start:
                # Flight phase: reach peak height - reference is elevated state
                ref_state = self.P_ref_state
            else:
                # Landing phase: return to ground - reference is terminal state
                ref_state = self.P_terminal_state

            # State tracking cost
            state_error_base = self.X[0:12, k] - ref_state[0:12]
            state_error_joint = self.X[12:24, k] - ref_state[12:24]

            # Input tracking cost
            input_error_vel = self.U[0:12, k] - self.P_ref_input[0:12]
            input_error_forces = self.U[12:24, k] - self.P_ref_input[12:24]

            # Quadratic costs
            cost += cs.mtimes([state_error_base.T, q_base, state_error_base])
            cost += cs.mtimes([state_error_joint.T, q_joint, state_error_joint])
            cost += cs.mtimes([input_error_vel.T, r_joint_vel, input_error_vel])
            cost += cs.mtimes([input_error_forces.T, r_forces, input_error_forces])

        # Terminal cost - use terminal state (landed position) as reference
        terminal_error_base = self.X[0:12, self.horizon] - self.P_terminal_state[0:12]
        terminal_error_joint = (
            self.X[12:24, self.horizon] - self.P_terminal_state[12:24]
        )

        # Higher weight on terminal cost to ensure proper landing
        cost += 2.0 * cs.mtimes(
            [terminal_error_base.T, q_terminal_base, terminal_error_base]
        )
        cost += 2.0 * cs.mtimes(
            [terminal_error_joint.T, q_terminal_joint, terminal_error_joint]
        )

        # Set objective
        self.opti.minimize(cost)

    def _setup_path_constraints(self) -> None:
        """Setup path constraints including friction cone, foot height, etc."""
        # Begin imposing path constraints only after the first timestep
        # This prevents conflicts with the initial state constraints
        for k in range(1, self.horizon):  # Start from k=1 instead of k=0
            contact_k = self.P_contact[:, k]
            u_k = self.U[:, k]
            x_k = self.X[:, k]

            for constraint in self.config.mpc_config.path_constraints:
                # Try new signature with k and horizon first, fall back to old signature
                try:
                    constraint_expr, constraint_l, constraint_u = constraint(
                        x_k,
                        u_k,
                        self.kindyn_model,
                        self.config,
                        contact_k,
                        k,
                        self.horizon,
                    )
                except TypeError:
                    # Fall back to old 5-argument signature for backward compatibility
                    constraint_expr, constraint_l, constraint_u = constraint(
                        x_k, u_k, self.kindyn_model, self.config, contact_k
                    )
                self.opti.subject_to(constraint_expr >= constraint_l)
                self.opti.subject_to(constraint_expr <= constraint_u)

        # NOTE: Terminal landing constraints have been removed from the base MPC.
        # The LLM should specify task-specific terminal constraints.
        # For a backflip, terminal pitch needs to be ~2π, not constrained to ±0.2 rad.
        # For a simple jump, the LLM can add terminal upright constraints.

    def _setup_solver(self) -> None:
        """Setup the solver options."""
        self.opti.solver("ipopt", self.config.solver_config)

    def solve_trajectory(
        self,
        initial_state: np.ndarray,
        ref: np.ndarray,
        contact_sequence: np.ndarray,
        warmstart: dict[str, Any] | None = None,
        ref_trajectory: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Solve the trajectory optimization problem.

        Args:
            initial_state: Initial state vector
            ref: Reference trajectory (shape: states_dim + inputs_dim)
            contact_sequence: Contact sequence array (shape: 4 x horizon)
            warmstart: Optional dict with 'X' and 'U' arrays from a previous solution
            ref_trajectory: Optional dict with 'X_ref' (states_dim, horizon+1) and
                'U_ref' (inputs_dim, horizon) used as solver initial guess.

        Returns:
            Tuple of (state_traj, grf_traj, joint_vel_traj, status)
        """
        # Set parameter values
        self.opti.set_value(self.P_initial_state, initial_state)
        self.opti.set_value(self.P_ref_state, ref[: self.states_dim])
        self.opti.set_value(self.P_ref_input, ref[self.states_dim :])
        self.opti.set_value(self.P_contact, contact_sequence)

        # Create terminal state for landing - should return to near-initial configuration
        terminal_state = initial_state.copy()
        terminal_state[0] = ref[0]  # X position from reference (forward motion)
        terminal_state[1] = ref[1]  # Y position from reference
        terminal_state[2] = initial_state[2]  # Z back to initial height
        terminal_state[3:6] = 0.0  # Linear velocity
        terminal_state[9:12] = 0.0  # Angular velocity
        terminal_state[12:24] = initial_state[12:24]  # Joint positions
        self.opti.set_value(self.P_terminal_state, terminal_state)

        # Robot parameters
        self.opti.set_value(self.P_mu, self.config.experiment.mu_ground)
        self.opti.set_value(self.P_mass, self.config.robot_data.mass)
        self.opti.set_value(self.P_inertia, self.config.robot_data.inertia.flatten())

        # Determine initial guess priority: warmstart > ref_trajectory > heuristic
        used_warmstart = False
        if warmstart is not None:
            ws_X = _interpolate_warmstart(
                warmstart.get("X"), self.states_dim, self.horizon + 1
            )
            ws_U = _interpolate_warmstart(
                warmstart.get("U"), self.inputs_dim, self.horizon
            )
            if ws_X is not None and ws_U is not None:
                prev_horizon = (
                    warmstart["X"].shape[1] - 1
                    if warmstart.get("X") is not None
                    else "?"
                )
                logger.info(
                    f"Warm-starting from previous solution "
                    f"(horizon {prev_horizon} → {self.horizon})"
                )
                X_init = ws_X
                U_init = ws_U
                used_warmstart = True

        if not used_warmstart and ref_trajectory is not None:
            # Use LLM-generated reference trajectory as initial guess
            logger.info("Using reference trajectory as initial guess")
            X_init = ref_trajectory["X_ref"].copy()
            U_init = ref_trajectory["U_ref"].copy()
        elif not used_warmstart:
            logger.info("Cold-starting with heuristic initial guess")
            # Better initial guess with phase-aware trajectory
            X_init = np.zeros((self.states_dim, self.horizon + 1))
            dt = self.config.mpc_config.mpc_dt

            for i in range(self.horizon + 1):
                X_init[:, i] = initial_state.copy()

                # Phase-aware height profile
                if i < self.pre_flight_steps:
                    X_init[2, i] = initial_state[2]
                elif i < self.landing_start:
                    flight_progress = (i - self.pre_flight_steps) / self.flight_steps
                    peak_height = ref[2]
                    height_offset = peak_height - initial_state[2]
                    X_init[2, i] = initial_state[
                        2
                    ] + height_offset * 4 * flight_progress * (1 - flight_progress)
                    X_init[5, i] = height_offset * 4 * (1 - 2 * flight_progress) / dt
                else:
                    X_init[2, i] = initial_state[2]
                    X_init[5, i] = 0.0

                forward_target = ref[0]
                progress = i / self.horizon
                X_init[0, i] = (
                    initial_state[0] + (forward_target - initial_state[0]) * progress
                )
                X_init[12:24, i] = initial_state[12:24]

            # Heuristic initial guess for inputs with phase awareness
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
                        if i >= self.landing_start:
                            U_init[12 + foot * 3 + 2, i] = (
                                self.config.robot_data.mass * 9.81 / 4 * 1.2
                            )
                        else:
                            U_init[12 + foot * 3 + 2, i] = (
                                self.config.robot_data.mass * 9.81 / 4 * 0.8
                            )
                    else:
                        U_init[12 + foot * 3 : 12 + foot * 3 + 3, i] = 0.0

        self.opti.set_initial(self.X, X_init)
        self.opti.set_initial(self.U, U_init)

        try:
            # Solve the optimization problem
            sol = self.opti.solve()

            # Extract solution
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)

            # Convert to expected format
            state_traj = X_opt.T  # Shape: (horizon+1, states_dim)
            joint_vel_traj = U_opt[0:12, :].T  # Shape: (horizon, 12)
            grf_traj = U_opt[12:24, :].T  # Shape: (horizon, 12)

            status = 0  # Success

        except Exception as e:
            print(f"Optimization failed: {e}")
            # Extract the infeasible trajectory using debug values
            # This gives us the solver's last iterate, useful for debugging
            try:
                X_debug = self.opti.debug.value(self.X)
                U_debug = self.opti.debug.value(self.U)
                state_traj = X_debug.T  # Shape: (horizon+1, states_dim)
                joint_vel_traj = U_debug[0:12, :].T  # Shape: (horizon, 12)
                grf_traj = U_debug[12:24, :].T  # Shape: (horizon, 12)
                print("📊 Extracted debug trajectory from failed optimization")
            except Exception:
                # Fall back to zeros if debug values not available
                state_traj = np.zeros((self.horizon + 1, self.states_dim))
                joint_vel_traj = np.zeros((self.horizon, 12))
                grf_traj = np.zeros((self.horizon, 12))
            status = 1  # Failure

        return state_traj, grf_traj, joint_vel_traj, status

    def get_constraint_violations(self) -> dict[str, Any]:
        """
        Analyze the current (possibly infeasible) iterate for debugging.

        Returns a dictionary with trajectory information useful for debugging.
        Note: Terminal constraints are LLM-specified, so we only report values, not violations.
        """
        info: dict[str, Any] = {
            "terminal_state": {},
            "trajectory_info": [],
            "summary": [],
        }

        try:
            # Get debug values (works even when solve failed)
            X_debug = self.opti.debug.value(self.X)

            # Report terminal state values (not violations - LLM decides what's valid)
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

            # Basic sanity checks (physics violations, not task-specific)
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
