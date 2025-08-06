import casadi as cs
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver

# Import the original kinodynamic model and configuration
from .model import KinoDynamic_Model


# New class for the hopping MPC, designed for a single trajectory optimization
class HoppingMPC:
    def __init__(self, model: KinoDynamic_Model, config, build=True):
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
        self._compile_and_initialize_solver(build)

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

        # Setup nonlinear constraints (total dimension: 20 + 4 + 12 = 36)
        self.nh = 36
        h_friction_cone, self.lb_friction_cone, self.ub_friction_cone = (
            self._friction_cone_constraints_expr(
                ocp,
                self.config.mpc_params["mu"],
                self.config.mpc_params["grf_min"],
                self.config.mpc_params["grf_max"],
            )
        )
        h_foot_height, lb_foot_height, ub_foot_height = (
            self._foot_height_constraints_expr(ocp)
        )
        h_foot_velocity, lb_foot_velocity, ub_foot_velocity = (
            self._foot_velocity_constraints_expr(ocp)
        )

        ocp.model.con_h_expr = cs.vertcat(
            h_friction_cone, h_foot_height, h_foot_velocity
        )
        ocp.constraints.lh = np.concatenate(
            (self.lb_friction_cone, lb_foot_height, lb_foot_velocity)
        )
        ocp.constraints.uh = np.concatenate(
            (self.ub_friction_cone, ub_foot_height, ub_foot_velocity)
        )

        # Setup the state and input box constraints
        ocp.constraints.x0 = np.zeros(self.states_dim)
        ocp.constraints.idxbu = np.arange(self.inputs_dim)
        ocp.constraints.lbu = np.array([-1e6] * self.inputs_dim)
        ocp.constraints.ubu = np.array([1e6] * self.inputs_dim)

        # Set the default parameter values (will take these values unless updated)
        # most of these are not used in our case, but we need to set them to something
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

    def _friction_cone_constraints_expr(
        self, ocp: AcadosOcp, mu: float, f_min: float, f_max: float
    ) -> (cs.SX, np.ndarray, np.ndarray):
        """
        Computes the friction cone constraints. Dimension: 20
        Adapted from the original centroidal NMPC problem
        """
        n = np.array([0, 0, 1])
        t = np.array([1, 0, 0])
        b = np.array([0, 1, 0])

        # Friction cone constraint patterns for each foot
        friction_cone_patterns = np.array(
            [
                -n * mu + t,  # Row 0: -μn + t
                -n * mu + b,  # Row 1: -μn + b
                n * mu + b,  # Row 2: μn + b
                n * mu + t,  # Row 3: μn + t
                n,  # Row 4: n
            ]
        )

        # Create the full constraint matrix using block structure
        Jbu = cs.SX.zeros(20, 12)
        for foot_idx in range(4):  # 4 feet
            row_start = foot_idx * 5
            col_start = foot_idx * 3
            for pattern_idx in range(5):  # 5 constraints per foot
                Jbu[row_start + pattern_idx, col_start : col_start + 3] = (
                    friction_cone_patterns[pattern_idx]
                )
        Jbu = Jbu @ cs.vertcat(self.kindyn_model.inputs[12:24])

        lbu = np.array([-1e6, -1e6, 0, 0, f_min] * 4)
        ubu = np.array([0, 0, 1e6, 1e6, f_max] * 4)
        return Jbu, lbu, ubu

    def _foot_height_constraints_expr(
        self, ocp: AcadosOcp
    ) -> (cs.SX, np.ndarray, np.ndarray):
        """
        Computes the height of the feet. Dimension: 4
        """
        foot_height_FL = self.kindyn_model.foot_position_fl[2]
        foot_height_FR = self.kindyn_model.foot_position_fr[2]
        foot_height_RL = self.kindyn_model.foot_position_rl[2]
        foot_height_RR = self.kindyn_model.foot_position_rr[2]
        return (
            cs.vertcat(foot_height_FL, foot_height_FR, foot_height_RL, foot_height_RR),
            np.zeros(4),
            np.ones(4) * 1e6,
        )

    def _foot_velocity_constraints_expr(
        self, ocp: AcadosOcp
    ) -> (cs.SX, np.ndarray, np.ndarray):
        """
        Computes the velocity of the footholds. Dimension: 12
        """
        qvel_joints_FL = self.kindyn_model.inputs[0:3]
        qvel_joints_FR = self.kindyn_model.inputs[3:6]
        qvel_joints_RL = self.kindyn_model.inputs[6:9]
        qvel_joints_RR = self.kindyn_model.inputs[9:12]

        joint_position = self.kindyn_model.states[12:24]
        com_position = self.kindyn_model.states[0:3]
        com_velocity = self.kindyn_model.states[3:6]
        com_angular_velocity = self.kindyn_model.states[9:12]
        roll = self.kindyn_model.states[6]
        pitch = self.kindyn_model.states[7]
        yaw = self.kindyn_model.states[8]
        b_R_w = self.kindyn_model.compute_b_R_w(roll, pitch, yaw)
        H = cs.SX.eye(4)
        H[0:3, 0:3] = b_R_w.T
        H[0:3, 3] = com_position

        qvel = cs.vertcat(
            com_velocity,
            com_angular_velocity,
            qvel_joints_FL,
            qvel_joints_FR,
            qvel_joints_RL,
            qvel_joints_RR,
        )

        foot_vel_FL = (
            self.kindyn_model.jacobian_FL_fun(H, joint_position)[0:3, :] @ qvel
        )  # qvel_FL
        foot_vel_FR = (
            self.kindyn_model.jacobian_FR_fun(H, joint_position)[0:3, :] @ qvel
        )  # qvel_FR
        foot_vel_RL = (
            self.kindyn_model.jacobian_RL_fun(H, joint_position)[0:3, :] @ qvel
        )  # qvel_RL
        foot_vel_RR = (
            self.kindyn_model.jacobian_RR_fun(H, joint_position)[0:3, :] @ qvel
        )  # qvel_RR

        return (
            cs.vertcat(foot_vel_FL, foot_vel_FR, foot_vel_RL, foot_vel_RR),
            -np.ones(12) * 1e6,
            np.ones(12) * 1e6,
        )

    def _compile_and_initialize_solver(self, build=True):
        """
        Compiles the solver and initializes the decision variables with zeros.
        """
        code_export_dir = self.config.mpc_params["compile_dir"]
        json_filename = code_export_dir + "/hopping_nmpc" + ".json"
        self.ocp.code_export_directory = str(code_export_dir)

        # Compile the solver
        self.acados_ocp_solver = AcadosOcpSolver(
            self.ocp,
            json_file=json_filename,
            verbose=False,
            build=build,
            generate=build,
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
        self.acados_ocp_solver.set(self.horizon, "yref", ref[:24])

    def _set_parameter_values(self, contact_sequence: np.ndarray) -> None:
        """
        Sets the parameter values for the OCP.
        """
        param_values = self.ocp.parameter_values.copy()
        for j in range(self.horizon):
            param_values[0:4] = contact_sequence[:, j]
            self.acados_ocp_solver.set(j, "p", param_values)

    def _compute_foot_height_constraints(
        self, contact_sequence: np.ndarray, EPS: float = 1e-3
    ) -> None:
        """
        Computes the bounds of the foot height constraints for the OCP.
        If a foot is in stance, the height constraint is lb=0, ub=EPS.
        If a foot is in swing, the height constraint is lb=EPS, ub=1e6.
        Returns:
            lb_foot_height: np.ndarray, shape (horizon, 4)
            ub_foot_height: np.ndarray, shape (horizon, 4)
        """
        lb_foot_height = np.zeros((self.horizon, 4))
        ub_foot_height = np.zeros((self.horizon, 4))
        for t in range(self.horizon):
            for i in range(4):
                if contact_sequence[i, t] == 1:
                    lb_foot_height[t, i] = 0
                    # DEBUG: deactivate this height constraint
                    # ub_foot_height[t, i] = EPS
                    ub_foot_height[t, i] = 0.1
                else:
                    lb_foot_height[t, i] = 0.0
                    ub_foot_height[t, i] = 1e6
        return lb_foot_height, ub_foot_height

    def _compute_foot_velocity_constraints(self, contact_sequence: np.ndarray) -> None:
        """
        Computes the bounds of the foot velocity constraints for the OCP.
        If a foot is in stance, the velocity constraint is lb=0, ub=0.
        If a foot is in swing, the velocity constraint is lb=-1e3, ub=1e3.
        Returns:
            lb_foot_velocity: np.ndarray, shape (horizon, 12)
            ub_foot_velocity: np.ndarray, shape (horizon, 12)
        """
        lb_foot_velocity = np.zeros((self.horizon, 12))
        ub_foot_velocity = np.zeros((self.horizon, 12))
        for t in range(self.horizon):
            for i in range(4):
                if contact_sequence[i, t] == 1:
                    # DEBUG: deactivate this velocity constraint
                    # lb_foot_velocity[t, i * 3 : (i + 1) * 3] = 0
                    # ub_foot_velocity[t, i * 3 : (i + 1) * 3] = 0
                    lb_foot_velocity[t, i * 3 : (i + 1) * 3] = -1.0
                    ub_foot_velocity[t, i * 3 : (i + 1) * 3] = 1.0
                else:
                    lb_foot_velocity[t, i * 3 : (i + 1) * 3] = -1e3
                    ub_foot_velocity[t, i * 3 : (i + 1) * 3] = 1e3
        return lb_foot_velocity, ub_foot_velocity

    def _compute_friction_cone_constraints(self, contact_sequence: np.ndarray) -> None:
        """
        Computes the bounds of the friction cone constraints for the OCP.
        """
        lb_friction_cone = np.tile(self.lb_friction_cone, (self.horizon, 1))
        ub_friction_cone = np.tile(self.ub_friction_cone, (self.horizon, 1))
        for t in range(self.horizon):
            for i in range(4):
                if contact_sequence[i, t] == 0:
                    lb_friction_cone[t, i * 5 : (i + 1) * 5] = -1e6
                    ub_friction_cone[t, i * 5 : (i + 1) * 5] = 1e6
        return lb_friction_cone, ub_friction_cone

    def _set_constraints(self, contact_sequence: np.ndarray) -> None:
        """
        Sets the constraints for the OCP.
        """
        lb_foot_height, ub_foot_height = self._compute_foot_height_constraints(
            contact_sequence
        )
        lb_foot_velocity, ub_foot_velocity = self._compute_foot_velocity_constraints(
            contact_sequence
        )
        lb_friction_cone, ub_friction_cone = self._compute_friction_cone_constraints(
            contact_sequence
        )
        lh = np.hstack((lb_friction_cone, lb_foot_height, lb_foot_velocity))
        uh = np.hstack((ub_friction_cone, ub_foot_height, ub_foot_velocity))

        # Finally, also add an actuation limit constraint
        lbu = np.concatenate((np.ones(12) * -1e2, np.ones(12) * -1e2))
        ubu = np.concatenate((np.ones(12) * 1e2, np.ones(12) * 1e2))

        for t in range(1, self.horizon):
            self.acados_ocp_solver.constraints_set(t, "lh", lh[t, :])
            self.acados_ocp_solver.constraints_set(t, "uh", uh[t, :])
            self.acados_ocp_solver.set(t, "lbu", lbu)
            self.acados_ocp_solver.set(t, "ubu", ubu)

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
        self._set_parameter_values(contact_sequence)
        self._set_constraints(contact_sequence)

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
