import casadi as cs
import numpy as np
from liecasadi import SO3

from .constraints import constraints as constr
from .dynamics.model import KinoDynamic_Model


class QuadrupedMPCOpti:
    """
    Hopping MPC implementation using CasADi's Opti framework.

    This class transcribes the Acados-based trajectory optimization into
    a more intuitive Opti formulation that mirrors the mathematical structure
    of the optimization problem directly.
    """

    def __init__(self, model: KinoDynamic_Model, config, build=True):
        """
        Initializes the hopping MPC solver using CasADi Opti.

        Args:
            model: KinoDynamic_Model instance
            config: Configuration object with MPC parameters
            build: Whether to build the solver (kept for compatibility)
        """
        self.horizon = config.mpc_params["horizon"]
        self.config = config
        self.kindyn_model = model

        # Get dimensions from the kinodynamic model
        acados_model = self.kindyn_model.export_robot_model()
        self.states_dim = acados_model.x.size()[0]
        self.inputs_dim = acados_model.u.size()[0]

        # Initialize the Opti optimization environment
        self.opti = cs.Opti()

        # Create decision variables
        self._create_decision_variables()

        # Setup the optimization problem structure
        self._setup_optimization_problem()

    def _create_decision_variables(self):
        """Create the decision variables for the trajectory optimization."""
        # State trajectory: (horizon+1, states_dim)
        self.X = self.opti.variable(self.states_dim, self.horizon + 1)

        # Input trajectory: (horizon, inputs_dim)
        self.U = self.opti.variable(self.inputs_dim, self.horizon)

        # Parameters that can be set at runtime
        self.P_contact = self.opti.parameter(4, self.horizon)  # Contact sequence
        self.P_ref_state = self.opti.parameter(self.states_dim)  # Reference state
        self.P_ref_input = self.opti.parameter(self.inputs_dim)  # Reference input
        self.P_initial_state = self.opti.parameter(self.states_dim)  # Initial state

        # Robot parameters
        self.P_mu = self.opti.parameter()  # Friction coefficient
        self.P_grf_min = self.opti.parameter()  # Min ground reaction force
        self.P_grf_max = self.opti.parameter()  # Max ground reaction force
        self.P_mass = self.opti.parameter()  # Robot mass
        self.P_inertia = self.opti.parameter(9)  # Flattened inertia matrix

    def _setup_optimization_problem(self):
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

    def _setup_dynamics_constraints(self):
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
            dt = self.config.mpc_params["dt"]
            f_k = self._dynamics_fun(x_k, u_k, param_k)

            # Integration constraint (explicit Euler for now)
            self.opti.subject_to(x_next == x_k + dt * f_k)

    def _create_dynamics_function(self):
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

    def _setup_cost_function(self):
        """Setup the quadratic tracking cost function."""
        # Cost weights from config
        q_base = self.config.mpc_params["q_base"]
        q_joint = self.config.mpc_params["q_joint"]
        r_joint_vel = self.config.mpc_params["r_joint_vel"]
        r_forces = self.config.mpc_params["r_forces"]
        q_terminal_base = self.config.mpc_params["q_terminal_base"]
        q_terminal_joint = self.config.mpc_params["q_terminal_joint"]

        # Initialize cost
        cost = 0

        # Stage costs (intermediate stages)
        for k in range(self.horizon):
            # State tracking cost
            state_error_base = self.X[0:12, k] - self.P_ref_state[0:12]
            state_error_joint = self.X[12:24, k] - self.P_ref_state[12:24]

            # Input tracking cost
            input_error_vel = self.U[0:12, k] - self.P_ref_input[0:12]
            input_error_forces = self.U[12:24, k] - self.P_ref_input[12:24]

            # Quadratic costs
            cost += cs.mtimes([state_error_base.T, q_base, state_error_base])
            cost += cs.mtimes([state_error_joint.T, q_joint, state_error_joint])
            cost += cs.mtimes([input_error_vel.T, r_joint_vel, input_error_vel])
            cost += cs.mtimes([input_error_forces.T, r_forces, input_error_forces])

        # Terminal cost
        terminal_error_base = self.X[0:12, self.horizon] - self.P_ref_state[0:12]
        terminal_error_joint = self.X[12:24, self.horizon] - self.P_ref_state[12:24]

        cost += cs.mtimes([terminal_error_base.T, q_terminal_base, terminal_error_base])
        cost += cs.mtimes(
            [terminal_error_joint.T, q_terminal_joint, terminal_error_joint]
        )

        # Set objective
        self.opti.minimize(cost)

    def _setup_path_constraints(self):
        """Setup path constraints including friction cone, foot height, etc."""
        for k in range(self.horizon):
            contact_k = self.P_contact[:, k]
            u_k = self.U[:, k]
            x_k = self.X[:, k]

            for constraint in self.config.path_constraints:
                constraint_expr, constraint_l, constraint_u = constraint(
                    x_k, u_k, self.kindyn_model, self.config, contact_k
                )
                self.opti.subject_to(constraint_expr >= constraint_l)
                self.opti.subject_to(constraint_expr <= constraint_u)

    def _setup_solver(self):
        """Setup the solver options."""
        # Use IPOPT solver with valid settings
        opts = {
            "ipopt.print_level": 5,
            "print_time": True,
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.mu_init": 1e-2,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.alpha_for_y": "primal",
            "ipopt.recalc_y": "yes",
        }
        self.opti.solver("ipopt", opts)

    def solve_trajectory(
        self, initial_state: dict, ref: np.ndarray, contact_sequence: np.ndarray
    ) -> tuple:
        """
        Solve the trajectory optimization problem.

        Args:
            initial_state: Dictionary with initial state components
            ref: Reference trajectory (shape: states_dim + inputs_dim)
            contact_sequence: Contact sequence array (shape: 4 x horizon)

        Returns:
            Tuple of (state_traj, grf_traj, joint_vel_traj, status)
        """
        # Set initial state
        state_acados = np.concatenate(
            [
                initial_state["position"],
                initial_state["linear_velocity"],
                initial_state["orientation"],
                initial_state["angular_velocity"],
                initial_state["joint_FL"],
                initial_state["joint_FR"],
                initial_state["joint_RL"],
                initial_state["joint_RR"],
                np.zeros(6),  # Integral states
            ]
        )

        # Set parameter values
        self.opti.set_value(self.P_initial_state, state_acados)
        self.opti.set_value(self.P_ref_state, ref[: self.states_dim])
        self.opti.set_value(self.P_ref_input, ref[self.states_dim :])
        self.opti.set_value(self.P_contact, contact_sequence)

        # Robot parameters
        self.opti.set_value(self.P_mu, self.config.mpc_params["mu"])
        self.opti.set_value(self.P_grf_min, self.config.mpc_params["grf_min"])
        self.opti.set_value(self.P_grf_max, self.config.mpc_params["grf_max"])
        self.opti.set_value(self.P_mass, self.config.robot_data.mass)
        self.opti.set_value(self.P_inertia, self.config.robot_data.inertia.flatten())

        # Better initial guess for joint positions
        # Start with the initial configuration and keep it reasonable
        X_init = np.zeros((self.states_dim, self.horizon + 1))
        for i in range(self.horizon + 1):
            X_init[:, i] = state_acados.copy()
            # Ensure joint angles stay in reasonable ranges
            X_init[12:24, i] = state_acados[12:24]  # Keep initial joint configuration

        self.opti.set_initial(self.X, X_init)

        # Better initial guess for inputs
        U_init = np.zeros((self.inputs_dim, self.horizon))
        for i in range(self.horizon):
            # Very small joint velocities
            U_init[0:12, i] = 0.001 * np.sin(
                np.arange(12) * 0.1
            )  # Much smaller velocities

            # Gravity compensation forces for stance feet (more conservative)
            contact_i = (
                contact_sequence[:, i]
                if i < contact_sequence.shape[1]
                else contact_sequence[:, -1]
            )
            for foot in range(4):
                if contact_i[foot] > 0.5:  # In stance
                    # More conservative force distribution
                    U_init[12 + foot * 3 + 2, i] = (
                        self.config.robot_data.mass * 9.81 / 4 * 0.8
                    )  # 80% of weight
                else:
                    # No forces during flight
                    U_init[12 + foot * 3 : 12 + foot * 3 + 3, i] = 0.0

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
            # Return empty trajectories on failure
            state_traj = np.zeros((self.horizon + 1, self.states_dim))
            joint_vel_traj = np.zeros((self.horizon, 12))
            grf_traj = np.zeros((self.horizon, 12))
            status = 1  # Failure

        return state_traj, grf_traj, joint_vel_traj, status
