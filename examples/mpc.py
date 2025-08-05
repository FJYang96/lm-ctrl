import casadi as cs
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver

# Import the original kinodynamic model and configuration
from .model import KinoDynamic_Model


# New class for the hopping MPC, designed for a single trajectory optimization
class HoppingMPC:
    def __init__(self, model: KinoDynamic_Model, config):
        """
        Initializes and compiles the hopping MPC solver.
        """
        self.horizon = config.mpc_params["horizon"]
        self.config = config

        # Create the class of the kinodynamic model and instantiate the acados model
        self.kindyn_model = model
        acados_model = self.kindyn_model.export_robot_model()
        self.states_dim = acados_model.x.size()[0]
        self.inputs_dim = acados_model.u.size()[0]

        # Create the acados ocp solver
        self.ocp = self._create_ocp_solver_description(acados_model)
        self._compile_and_initialize_solver()

    def _create_ocp_solver_description(self, acados_model) -> AcadosOcp:
        """
        Sets up the Acados OCP object with cost, constraints, and solver options.
        These will be updated via `self._setup_cost` and `self._setup_constraints`.
        """
        ocp = AcadosOcp()
        ocp.model = acados_model

        # Specify the horizon and dimensions of the OCP
        ocp.dims.N = self.horizon
        ocp.dims.N_horizon = self.horizon
        ocp.dims.nbu = self.inputs_dim

        # Setup the form of the quadratic tracking cost function
        self._setup_cost(
            ocp,
            self.config.mpc_params["q_base"],
            self.config.mpc_params["q_joint"],
            self.config.mpc_params["r_joint_vel"],
            self.config.mpc_params["r_forces"],
            self.config.mpc_params["q_terminal_base"],
            self.config.mpc_params["q_terminal_joint"],
        )

        # Setup the form of the state constraint
        ocp.constraints.x0 = np.zeros(self.states_dim)
        ocp.constraints.idxbu = np.arange(self.inputs_dim)
        ocp.constraints.lbu = np.array([-1e6] * self.inputs_dim)
        ocp.constraints.ubu = np.array([1e6] * self.inputs_dim)

        # TODO: why are the base position and yaw not updated?
        # Set the default parameter values (will take these values unless updated)
        init_contact_status = np.array([1.0, 1.0, 1.0, 1.0])
        init_mu = np.array([self.config.mpc_params["mu"]])
        init_stance_proximity = np.array([0, 0, 0, 0])
        init_base_position = np.array([0, 0, 0])
        init_base_yaw = np.array([0])
        init_external_wrench = np.array([0, 0, 0, 0, 0, 0])
        init_inertia = self.config.inertia.flatten()
        init_mass = np.array([self.config.mass])

        ocp.parameter_values = np.concatenate(
            (
                init_contact_status,
                init_mu,
                init_stance_proximity,
                init_base_position,
                init_base_yaw,
                init_external_wrench,
                init_inertia,
                init_mass,
            )
        )

        # Solver options
        ocp.solver_options.qp_solver = self.config.mpc_params["qp_solver"]
        ocp.solver_options.hessian_approx = self.config.mpc_params["hessian_approx"]
        ocp.solver_options.integrator_type = self.config.mpc_params["integrator_type"]
        ocp.solver_options.nlp_solver_type = self.config.mpc_params["nlp_solver_type"]
        ocp.solver_options.nlp_solver_max_iter = self.config.mpc_params[
            "nlp_solver_max_iter"
        ]
        ocp.solver_options.tf = self.config.duration

        return ocp

    def _setup_cost(
        self,
        ocp: AcadosOcp,
        q_base: np.ndarray = np.eye(12) * 1e-2,
        q_joint: np.ndarray = np.eye(12) * 1e-2,
        r_joint_vel: np.ndarray = np.eye(12) * 1e-6,
        r_forces: np.ndarray = np.eye(12) * 1e-8,
        q_terminal_base: np.ndarray = np.eye(12) * 1e-1,
        q_terminal_joint: np.ndarray = np.eye(12) * 1e-1,
    ) -> None:
        """Sets up the cost function for the OCP.
        The cost function is a quadratic cost function of the form:
            J(x, u) = 1/2 * (y - yref) ^T * W * (y - yref)
        where y is a function of x and u. Specifically, for the intermediate stages,
            y = [base_states, joint_positions, joint_velocities, GRFs]
        and for the terminal stage,
            y = [base_states, joint_positions]
        """
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        nx = self.states_dim
        nu = self.inputs_dim

        # The cost expression for intermediate stages
        ocp.model.cost_y_expr = cs.vertcat(
            ocp.model.x[0:12],  # Base states
            ocp.model.x[12:24],  # Joint positions
            ocp.model.u,  # Inputs (joint velocities and GRFs)
        )
        ny = ocp.model.cost_y_expr.size()[0]  # This will be 12 + 12 + 24 = 48

        # The cost expression for the terminal stage
        ocp.model.cost_y_expr_e = cs.vertcat(
            ocp.model.x[0:12],  # Base states
            ocp.model.x[12:24],  # Joint positions
        )
        ny_e = ocp.model.cost_y_expr_e.size()[0]  # This will be 12 + 12 = 24

        # Weight matrices for the quadratic cost function
        ocp.cost.W = scipy.linalg.block_diag(
            np.eye(12) * q_base,
            np.eye(12) * q_joint,
            np.eye(12) * r_joint_vel,
            np.eye(12) * r_forces,
        )
        ocp.cost.W_e = scipy.linalg.block_diag(
            np.eye(12) * q_terminal_base,
            np.eye(12) * q_terminal_joint,
        )

        # Define Vx and Vu to map x and u to y_expr, i.e. y = Vx @ x + Vu @ u
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vx[:24, :] = np.eye(24, nx)
        ocp.cost.Vu[24:, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(ny_e, nx)  # Map final states to terminal cost

        # Initialize yref and yref_e to zero
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

    def _compile_and_initialize_solver(self):
        """
        Compiles the solver and initializes the decision variables with zeros.
        """
        code_export_dir = self.config.mpc_params["compile_dir"]
        json_filename = code_export_dir + "/hopping_nmpc" + ".json"
        self.ocp.code_export_directory = str(code_export_dir)

        # Compile the solver
        self.acados_ocp_solver = AcadosOcpSolver(
            self.ocp, json_file=json_filename, verbose=False
        )

        # Initialize the solutions with zeros
        for stage in range(self.horizon + 1):
            self.acados_ocp_solver.set(stage, "x", np.zeros((self.states_dim,)))
        for stage in range(self.horizon):
            self.acados_ocp_solver.set(stage, "u", np.zeros((self.inputs_dim,)))

    def _set_initial_state(self, initial_state: dict) -> None:
        """
        Sets the initial state constraint.
        """
        state_acados = np.concatenate(
            (
                initial_state["position"],
                initial_state["linear_velocity"],
                initial_state["orientation"],
                initial_state["angular_velocity"],
                initial_state["joint_FL"],
                initial_state["joint_FR"],
                initial_state["joint_RL"],
                initial_state["joint_RR"],
                np.zeros(6),  # For the integral states
            )
        )
        self.acados_ocp_solver.set(0, "lbx", state_acados)
        self.acados_ocp_solver.set(0, "ubx", state_acados)

    def _set_reference(self, ref: np.ndarray) -> None:
        """
        Sets the reference for the OCP.
        """
        # intermediate references
        for j in range(self.horizon):
            self.acados_ocp_solver.set(j, "yref", ref)
        # terminal reference
        self.acados_ocp_solver.set(self.horizon, "yref_e", ref[:24])

    def solve_trajectory(
        self,
        initial_state: dict,
        ref: np.ndarray,
        contact_sequence: np.ndarray,
    ) -> (np.ndarray, np.ndarray, np.ndarray, int):
        """
        Solves the trajectory optimization problem and returns the results.
        """
        self._set_initial_state(initial_state)
        self._set_reference(ref)

        for j in range(self.horizon):
            # Define lower and upper bounds for inputs (joint velocities, GRFs)
            lbu = np.array([-1e6] * self.inputs_dim)
            ubu = np.array([1e6] * self.inputs_dim)

            mpc_params = self.config.mpc_params
            # Contact forces (stance phase)
            if contact_sequence[0, j] == 1:  # FL foot is in stance
                lbu[12:15] = [
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_min"],
                ]
                ubu[12:15] = [
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_max"],
                ]
            else:  # FL foot is in flight
                lbu[12:15] = [0.0, 0.0, 0.0]
                ubu[12:15] = [0.0, 0.0, 0.0]

            # Repeat for other feet
            if contact_sequence[1, j] == 1:  # FR foot is in stance
                lbu[15:18] = [
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_min"],
                ]
                ubu[15:18] = [
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_max"],
                ]
            else:  # FR foot is in flight
                lbu[15:18] = [0.0, 0.0, 0.0]
                ubu[15:18] = [0.0, 0.0, 0.0]

            if contact_sequence[2, j] == 1:  # RL foot is in stance
                lbu[18:21] = [
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_min"],
                ]
                ubu[18:21] = [
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_max"],
                ]
            else:  # RL foot is in flight
                lbu[18:21] = [0.0, 0.0, 0.0]
                ubu[18:21] = [0.0, 0.0, 0.0]

            if contact_sequence[3, j] == 1:  # RR foot is in stance
                lbu[21:24] = [
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    -mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_min"],
                ]
                ubu[21:24] = [
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["mu"] * mpc_params["grf_max"],
                    mpc_params["grf_max"],
                ]
            else:  # RR foot is in flight
                lbu[21:24] = [0.0, 0.0, 0.0]
                ubu[21:24] = [0.0, 0.0, 0.0]

            # In the stance phase, joint velocities must be close to zero.
            if contact_sequence[:, j].any():
                lbu[0:12] = [-1e-2] * 12
                ubu[0:12] = [1e-2] * 12
            else:  # In flight, joint velocities can be anything
                lbu[0:12] = [-1e6] * 12
                ubu[0:12] = [1e6] * 12

            self.acados_ocp_solver.constraints_set(j, "lbu", lbu)
            self.acados_ocp_solver.constraints_set(j, "ubu", ubu)

            # --- FIX 6: Update the parameter values at each stage
            # The `kinodynamic_nmpc` also updates parameters per stage.
            param_values = self.ocp.parameter_values.copy()
            param_values[0:4] = contact_sequence[:, j]
            self.acados_ocp_solver.set(j, "p", param_values)

        # Solve the optimization problem
        status = self.acados_ocp_solver.solve()
        state_traj, grf_traj, joint_vel_traj = self._extract_trajectory()
        return state_traj, grf_traj, joint_vel_traj, status

    def _extract_trajectory(self):
        # Extract the full trajectory
        state_traj = np.zeros((self.horizon + 1, self.states_dim))
        inputs_traj = np.zeros((self.horizon, self.inputs_dim))

        for i in range(self.horizon + 1):
            state_traj[i, :] = self.acados_ocp_solver.get(i, "x")

        for i in range(self.horizon):
            inputs_traj[i, :] = self.acados_ocp_solver.get(i, "u")

        joint_vel_traj = inputs_traj[:, 0:12]
        grf_traj = inputs_traj[:, 12:24]

        return state_traj, grf_traj, joint_vel_traj
