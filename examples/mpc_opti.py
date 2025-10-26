import casadi as cs
import numpy as np
from liecasadi import SO3

from .model import KinoDynamic_Model


class HoppingMPCOpti:
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
        if acados_model is not None:
            self.states_dim = acados_model.x.size()[0]
            self.inputs_dim = acados_model.u.size()[0]
        else:
            # Fallback when acados is not available
            self.states_dim = self.kindyn_model.get_state_dim()
            self.inputs_dim = self.kindyn_model.get_input_dim()
        
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
        cost += cs.mtimes([terminal_error_joint.T, q_terminal_joint, terminal_error_joint])
        
        # Set objective
        self.opti.minimize(cost)
        
    def _setup_dynamics_constraints(self):
        """Setup dynamics constraints using the kinodynamic model."""
        # Create a CasADi function for dynamics that works with MX
        if not hasattr(self, '_dynamics_fun'):
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
                self.P_mass  # Mass
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
        
        if acados_model is not None:
            # Create the function directly from the SX expression
            # This automatically handles the conversion to work with different variable types
            return cs.Function('dynamics', 
                              [acados_model.x, acados_model.u, acados_model.p], 
                              [acados_model.f_expl_expr])
        else:
            # Fallback: create function using model's symbolic variables directly
            # Build the parameter vector the same way as in export_robot_model
            param = cs.vertcat(
                self.kindyn_model.stance_param,
                self.kindyn_model.mu_friction,
                self.kindyn_model.stance_proximity,
                self.kindyn_model.base_position,
                self.kindyn_model.base_yaw,
                self.kindyn_model.external_wrench,
                self.kindyn_model.inertia,
                self.kindyn_model.mass,
            )
            f_expl = self.kindyn_model.forward_dynamics(self.kindyn_model.states, self.kindyn_model.inputs, param)
            return cs.Function('dynamics', 
                              [self.kindyn_model.states, self.kindyn_model.inputs, param], 
                              [f_expl])
            
    def _setup_path_constraints(self):
        """Setup path constraints including friction cone, foot height, etc."""
        for k in range(self.horizon):
            contact_k = self.P_contact[:, k]
            u_k = self.U[:, k]
            x_k = self.X[:, k]
            
            # Friction cone constraints
            self._add_friction_cone_constraints(u_k, contact_k, k)
            
            # Foot height constraints
            self._add_foot_height_constraints(x_k, contact_k, k)
            
            # Foot velocity constraints  
            self._add_foot_velocity_constraints(x_k, u_k, contact_k, k)
            
            # Add joint limits to prevent broken configurations
            joint_positions = x_k[12:24]  # 12 joint angles
            
            # Basic joint limits for Go2 robot (approximate)
            # Hip: ±45°, Thigh: -90° to +90°, Calf: -150° to -30°
            joint_limits_lower = np.array([
                -0.8, -1.6, -2.6,  # FL: hip, thigh, calf
                -0.8, -1.6, -2.6,  # FR
                -0.8, -1.6, -2.6,  # RL  
                -0.8, -1.6, -2.6,  # RR
            ])
            joint_limits_upper = np.array([
                0.8, 1.6, -0.5,   # FL: hip, thigh, calf
                0.8, 1.6, -0.5,   # FR
                0.8, 1.6, -0.5,   # RL
                0.8, 1.6, -0.5,   # RR  
            ])
            
            # Apply joint limits
            for joint_idx in range(12):
                self.opti.subject_to(joint_positions[joint_idx] >= joint_limits_lower[joint_idx])
                self.opti.subject_to(joint_positions[joint_idx] <= joint_limits_upper[joint_idx])
            
            # Input bounds (already there but let's make them more reasonable)
            self.opti.subject_to(self.opti.bounded(-10, u_k[0:12], 10))  # Joint velocities (rad/s)
            self.opti.subject_to(self.opti.bounded(-500, u_k[12:24], 500))  # Forces (N)
            
    def _add_friction_cone_constraints(self, u_k, contact_k, k):
        """Add friction cone constraints exactly like working Acados version."""
        # Extract forces for each foot
        forces = u_k[12:24]
        
        for foot_idx in range(4):
            f_x = forces[foot_idx * 3]
            f_y = forces[foot_idx * 3 + 1] 
            f_z = forces[foot_idx * 3 + 2]
            contact_flag = contact_k[foot_idx]
            
            # Key insight: The working version uses contact_sequence to DISABLE constraints
            # When contact_flag = 0, ALL force constraints become [-1e6, 1e6] (no constraint)
            # When contact_flag = 1, proper friction cone constraints are applied
            
            # Normal force constraints
            self.opti.subject_to(f_z >= 0)  # Always non-negative
            
            # Contact-dependent force bounds
            # When in contact: proper bounds, when not in contact: very relaxed bounds
            f_max = self.P_grf_max
            large_bound = 1e6
            
            # Apply bounds that become very loose when not in contact
            self.opti.subject_to(f_x >= -large_bound * (1 - contact_flag) - f_max * contact_flag)
            self.opti.subject_to(f_x <= large_bound * (1 - contact_flag) + f_max * contact_flag)
            self.opti.subject_to(f_y >= -large_bound * (1 - contact_flag) - f_max * contact_flag)
            self.opti.subject_to(f_y <= large_bound * (1 - contact_flag) + f_max * contact_flag)
            self.opti.subject_to(f_z <= large_bound * (1 - contact_flag) + f_max * contact_flag)
            
            # Friction cone constraints (only meaningful when in contact)
            mu_term = self.P_mu * f_z
            relaxation = large_bound * (1 - contact_flag)
            
            self.opti.subject_to(f_x <= mu_term + relaxation)
            self.opti.subject_to(f_x >= -mu_term - relaxation)
            self.opti.subject_to(f_y <= mu_term + relaxation)
            self.opti.subject_to(f_y >= -mu_term - relaxation)
            
    def _add_foot_height_constraints(self, x_k, contact_k, k):
        """Add very basic foot height constraints."""
        # For debugging: only apply the most basic ground penetration constraint
        # Extract components for forward kinematics
        com_position = x_k[0:3]
        roll = x_k[6]
        pitch = x_k[7] 
        yaw = x_k[8]
        joint_positions = x_k[12:24]
        
        # Create homogeneous transformation matrix
        from liecasadi import SO3
        w_R_b = SO3.from_euler(cs.vertcat(roll, pitch, yaw)).as_matrix()
        b_R_w = w_R_b.T
        H = cs.MX.eye(4)
        H[0:3, 0:3] = b_R_w.T
        H[0:3, 3] = com_position
        
        # Compute foot heights
        foot_height_fl = self.kindyn_model.forward_kinematics_FL_fun(H, joint_positions)[2, 3]
        foot_height_fr = self.kindyn_model.forward_kinematics_FR_fun(H, joint_positions)[2, 3]
        foot_height_rl = self.kindyn_model.forward_kinematics_RL_fun(H, joint_positions)[2, 3]
        foot_height_rr = self.kindyn_model.forward_kinematics_RR_fun(H, joint_positions)[2, 3]
        
        foot_heights = [foot_height_fl, foot_height_fr, foot_height_rl, foot_height_rr]
        
        # Only prevent severe ground penetration, no upper bounds for now
        for foot_idx in range(4):
            self.opti.subject_to(foot_heights[foot_idx] >= -0.05)  # Allow some penetration for numerical stability
            
    def _add_foot_velocity_constraints(self, x_k, u_k, contact_k, k):
        """Disable foot velocity constraints for debugging."""
        # Completely disable velocity constraints to isolate the issue
        pass

    def _setup_solver(self):
        """Setup the solver options."""
        # Use IPOPT solver with valid settings
        opts = {
            'ipopt.print_level': 5,
            'print_time': True,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.mu_init': 1e-2,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.alpha_for_y': 'primal',
            'ipopt.recalc_y': 'yes',
        }
        self.opti.solver('ipopt', opts)
        
    def solve_trajectory(self, initial_state: dict, ref: np.ndarray, 
                        contact_sequence: np.ndarray) -> tuple:
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
        state_acados = np.concatenate([
            initial_state["position"],
            initial_state["linear_velocity"], 
            initial_state["orientation"],
            initial_state["angular_velocity"],
            initial_state["joint_FL"],
            initial_state["joint_FR"],
            initial_state["joint_RL"],
            initial_state["joint_RR"],
            np.zeros(6),  # Integral states
        ])
        
        # Set parameter values
        self.opti.set_value(self.P_initial_state, state_acados)
        self.opti.set_value(self.P_ref_state, ref[:self.states_dim])
        self.opti.set_value(self.P_ref_input, ref[self.states_dim:])
        self.opti.set_value(self.P_contact, contact_sequence)
        
        # Robot parameters
        self.opti.set_value(self.P_mu, self.config.mpc_params["mu"])
        self.opti.set_value(self.P_grf_min, self.config.mpc_params["grf_min"])
        self.opti.set_value(self.P_grf_max, self.config.mpc_params["grf_max"])
        self.opti.set_value(self.P_mass, self.config.mass)
        self.opti.set_value(self.P_inertia, self.config.inertia.flatten())
        
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
            U_init[0:12, i] = 0.001 * np.sin(np.arange(12) * 0.1)  # Much smaller velocities
            
            # Gravity compensation forces for stance feet (more conservative)
            contact_i = contact_sequence[:, i] if i < contact_sequence.shape[1] else contact_sequence[:, -1]
            for foot in range(4):
                if contact_i[foot] > 0.5:  # In stance
                    # More conservative force distribution
                    U_init[12 + foot*3 + 2, i] = self.config.mass * 9.81 / 4 * 0.8  # 80% of weight
                else:
                    # No forces during flight
                    U_init[12 + foot*3:12 + foot*3 + 3, i] = 0.0
                    
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