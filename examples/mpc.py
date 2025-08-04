# Description: This file contains a new Acados NMPC controller class specifically
# designed for generating a hopping trajectory. It defines a single optimization
# problem for the entire hop, with different constraints for the stance and flight phases.

import pathlib
import copy
import os
import casadi as cs
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver

# Import the original kinodynamic model and configuration
from .model import KinoDynamic_Model

# New class for the hopping MPC, designed for a single trajectory optimization
class HoppingMPC:
    def __init__(self, T: float, dt: float, config):
        """
        Initializes the hopping MPC solver.

        Args:
            T (float): The total duration of the optimization horizon in seconds.
            dt (float): The duration of a single time step in seconds.
        """
        self.T = T
        self.dt = dt
        self.horizon = int(self.T / self.dt)
        self.config = config

        # Create the class of the kinodynamic model and instantiate the acados model
        self.kindyn_model = KinoDynamic_Model(config)
        acados_model = self.kindyn_model.export_robot_model()
        self.states_dim = acados_model.x.size()[0]
        self.inputs_dim = acados_model.u.size()[0]

        # Create the acados ocp solver
        self.ocp = self._create_ocp_solver_description(acados_model)

        # Set code export directory
        code_export_dir = pathlib.Path(__file__).parent.parent / "c_generated_code"
        self.ocp.code_export_directory = str(code_export_dir)

        # The acados_ocp_solver.py is now called with `acados_ocp` that has the
        # `parameter_values` set, which will fix the error.
        self.acados_ocp_solver = AcadosOcpSolver(
            self.ocp, json_file=self.ocp.code_export_directory + "/hopping_nmpc" + ".json"
        )

        # Initialize solver with zeros
        for stage in range(self.horizon + 1):
            self.acados_ocp_solver.set(stage, "x", np.zeros((self.states_dim,)))
        for stage in range(self.horizon):
            self.acados_ocp_solver.set(stage, "u", np.zeros((self.inputs_dim,)))

    def _create_ocp_solver_description(self, acados_model) -> AcadosOcp:
        """
        Sets up the Acados OCP object with cost, constraints, and solver options.
        """
        ocp = AcadosOcp()
        ocp.model = acados_model
        nx = self.states_dim
        nu = self.inputs_dim

        # --- FIX 1: Recalculate dimensions based on explicit expressions ---
        # The cost expression for intermediate stages
        cost_y_expr = cs.vertcat(
            acados_model.x[0:12], # Base states
            acados_model.x[12:24], # Joint positions
            acados_model.u # Inputs (joint velocities and GRFs)
        )
        ny = cost_y_expr.size()[0] # This will be 12 + 12 + 24 = 48

        # The cost expression for the terminal stage
        cost_y_expr_e = cs.vertcat(
            acados_model.x[0:12], # Base states
            acados_model.x[12:24], # Joint positions
        )
        ny_e = cost_y_expr_e.size()[0] # This will be 12 + 12 = 24

        ocp.dims.N = self.horizon
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        
        # --- FIX 2: Assign the symbolic cost expressions directly ---
        ocp.model.cost_y_expr = cost_y_expr
        ocp.model.cost_y_expr_e = cost_y_expr_e

        # Weight matrices
        Q_diag = np.diag([
            1e3, 1e3, 1e3,  # com_position
            1e2, 1e2, 1e2,  # com_velocity
            1e3, 1e3, 1e3,  # orientation_rpy
            1e1, 1e1, 1e1,  # angular_velocity
            1e-1, 1e-1, 1e-1, # joint_position FL
            1e-1, 1e-1, 1e-1, # joint_position FR
            1e-1, 1e-1, 1e-1, # joint_position RL
            1e-1, 1e-1, 1e-1, # joint_position RR
        ])
        R_diag_joint_vel = np.diag([1e-6] * 12)
        R_diag_forces = np.diag([1e-8] * 12)

        # --- FIX 3: Ensure W and W_e are sized correctly based on new dimensions ---
        ocp.cost.W = scipy.linalg.block_diag(Q_diag, R_diag_joint_vel, R_diag_forces)
        ocp.cost.W_e = Q_diag[0:ny_e, 0:ny_e] # Use a slice of Q_diag for the correct size

        # Define `Vx` and `Vu` to map `x` and `u` to `y_expr`
        ocp.cost.Vx = np.zeros((ny, nx))
        # The first `nx` terms of y_expr correspond to states
        ocp.cost.Vx[:24, :] = np.eye(nx)[:24, :]
        
        ocp.cost.Vu = np.zeros((ny, nu))
        # The last `nu` terms of y_expr correspond to inputs
        ocp.cost.Vu[24:, :] = np.eye(nu)

        ocp.cost.Vx_e = np.eye(ny_e, nx) # Map final states to terminal cost
        
        # --- FIX 4: Set the yref and yref_e sizes based on the new dimensions ---
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # Set initial state constraint
        ocp.constraints.x0 = np.zeros(nx)

        ocp.dims.nbu = nu
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.lbu = np.array([-1e6] * nu)
        ocp.constraints.ubu = np.array([1e6] * nu)

        # Set the initial parameter values
        init_contact_status = np.array([1.0, 1.0, 1.0, 1.0])
        init_mu = np.array([self.config.mpc_params['mu']])
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
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = self.config.mpc_params['num_qp_iterations']
        ocp.solver_options.tf = self.T

        return ocp

    def solve_trajectory(
        self,
        initial_state: dict,
        reference: dict,
        contact_sequence: np.ndarray,
    ) -> (np.ndarray, np.ndarray, np.ndarray, int):
        """
        Solves the trajectory optimization problem and returns the results.
        """
        # Set initial state constraint
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
                np.zeros(6), # For the integral states
            )
        )
        self.acados_ocp_solver.set(0, "lbx", state_acados)
        self.acados_ocp_solver.set(0, "ubx", state_acados)

        # --- FIX 5: Create and fill yref with the correct dimensions and values ---
        # The dimension of yref must match the symbolic cost expression.
        # It is: 12 (base states) + 12 (joint positions) + 24 (inputs) = 48.
        ny = 12 + 12 + 24
        
        for j in range(self.horizon):
            yref = np.zeros(ny)

            # Reference for base states (12 elements)
            yref[0:3] = reference["ref_position"]
            yref[3:6] = reference["ref_linear_velocity"]
            yref[6:9] = reference["ref_orientation"]
            yref[9:12] = reference["ref_angular_velocity"]
            
            # Reference for joint positions (12 elements)
            yref[12:24] = reference["ref_joints"]
            
            # Reference for inputs (12 joint velocities + 12 GRFs)
            # The default reference is zero for inputs.

            self.acados_ocp_solver.set(j, "yref", yref)

            # Define lower and upper bounds for inputs (joint velocities, GRFs)
            lbu = np.array([-1e6] * self.inputs_dim)
            ubu = np.array([1e6] * self.inputs_dim)

            mpc_params = self.config.mpc_params
            # Contact forces (stance phase)
            if contact_sequence[0, j] == 1: # FL foot is in stance
                lbu[12:15] = [-mpc_params['mu'] * mpc_params['grf_max'], -mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_min']]
                ubu[12:15] = [mpc_params['mu'] * mpc_params['grf_max'], mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_max']]
            else: # FL foot is in flight
                lbu[12:15] = [0.0, 0.0, 0.0]
                ubu[12:15] = [0.0, 0.0, 0.0]

            # Repeat for other feet
            if contact_sequence[1, j] == 1: # FR foot is in stance
                lbu[15:18] = [-mpc_params['mu'] * mpc_params['grf_max'], -mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_min']]
                ubu[15:18] = [mpc_params['mu'] * mpc_params['grf_max'], mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_max']]
            else: # FR foot is in flight
                lbu[15:18] = [0.0, 0.0, 0.0]
                ubu[15:18] = [0.0, 0.0, 0.0]

            if contact_sequence[2, j] == 1: # RL foot is in stance
                lbu[18:21] = [-mpc_params['mu'] * mpc_params['grf_max'], -mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_min']]
                ubu[18:21] = [mpc_params['mu'] * mpc_params['grf_max'], mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_max']]
            else: # RL foot is in flight
                lbu[18:21] = [0.0, 0.0, 0.0]
                ubu[18:21] = [0.0, 0.0, 0.0]

            if contact_sequence[3, j] == 1: # RR foot is in stance
                lbu[21:24] = [-mpc_params['mu'] * mpc_params['grf_max'], -mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_min']]
                ubu[21:24] = [mpc_params['mu'] * mpc_params['grf_max'], mpc_params['mu'] * mpc_params['grf_max'], mpc_params['grf_max']]
            else: # RR foot is in flight
                lbu[21:24] = [0.0, 0.0, 0.0]
                ubu[21:24] = [0.0, 0.0, 0.0]

            # In the stance phase, joint velocities must be close to zero.
            if contact_sequence[:, j].any():
                lbu[0:12] = [-1e-2] * 12
                ubu[0:12] = [1e-2] * 12
            else: # In flight, joint velocities can be anything
                lbu[0:12] = [-1e6] * 12
                ubu[0:12] = [1e6] * 12
            
            self.acados_ocp_solver.constraints_set(j, "lbu", lbu)
            self.acados_ocp_solver.constraints_set(j, "ubu", ubu)
            
            # --- FIX 6: Update the parameter values at each stage
            # The `kinodynamic_nmpc` also updates parameters per stage.
            param_values = self.ocp.parameter_values.copy()
            param_values[0:4] = contact_sequence[:, j]
            self.acados_ocp_solver.set(j, 'p', param_values)

        # --- FIX 7: Set final step reference to match new terminal dimension ---
        # The terminal cost expression is now size 24.
        ny_e = 12 + 12
        yref_e = np.zeros(ny_e)

        # Reference for final base states (12 elements)
        yref_e[0:3] = reference['ref_position']
        yref_e[3:6] = reference['ref_linear_velocity']
        yref_e[6:9] = reference['ref_orientation']
        yref_e[9:12] = reference['ref_angular_velocity']

        # Reference for final joint positions (12 elements)
        yref_e[12:24] = reference['ref_joints']

        self.acados_ocp_solver.set(self.horizon, "yref", yref_e)

        # Solve the optimization problem
        status = self.acados_ocp_solver.solve()

        # Extract the full trajectory
        state_traj = np.zeros((self.horizon + 1, self.states_dim))
        inputs_traj = np.zeros((self.horizon, self.inputs_dim))

        for i in range(self.horizon + 1):
            state_traj[i, :] = self.acados_ocp_solver.get(i, "x")

        for i in range(self.horizon):
            inputs_traj[i, :] = self.acados_ocp_solver.get(i, "u")

        joint_vel_traj = inputs_traj[:, 0:12]
        grf_traj = inputs_traj[:, 12:24]

        return state_traj, grf_traj, joint_vel_traj, status
