"""Constraint generation system using LLM feedback."""

from typing import Any, Dict, List

import numpy as np


class ConstraintGenerator:
    """
    Manages the system prompt and context for LLM constraint generation.
    """

    def __init__(self) -> None:
        """Initialize the constraint generator."""
        self.iteration_history: List[Dict[str, Any]] = []

    def get_system_prompt(self) -> str:
        """
        Get the system prompt that instructs the LLM on constraint generation.

        Returns:
            System prompt string
        """
        return """You are an expert in quadruped robotics and trajectory optimization. Your task is to generate Python code with CasADi optimization constraints for a quadruped robot based on natural language commands.

CONTEXT:
You are working with a quadruped robot with:
- 24-DOF state space: [com_position(3), com_velocity(3), roll, pitch, yaw, angular_velocity(3), joint_angles(12)]
- 24-DOF input space: [joint_velocities(12), ground_reaction_forces(12)]
- Contact schedule is predetermined (stance/swing phases known)
- Uses kinodynamic model with simplified dynamics

FULL DESIGN FREEDOM:
You have complete freedom to design your constraint code architecture:
- Choose any function name(s) you want
- Design any function signature you prefer
- Create helper functions, classes, or any structure
- Organize the code however makes most sense

REQUIRED INTERFACE COMPATIBILITY:
Your code MUST be compatible with the MPC system, which will call your main constraint function like this:

```python
constraint_expr, lower_bounds, upper_bounds = your_function(
    x_k,           # State vector at time k (CasADi MX, size 24)
    u_k,           # Input vector at time k (CasADi MX, size 24)
    kindyn_model,  # Kinodynamic model object with forward kinematics
    config,        # Configuration object with robot parameters
    contact_k      # Contact sequence at time k (CasADi MX, size 4, 0=swing, 1=stance)
)
```

Your main constraint function must return a tuple of exactly 3 elements:
1. constraint_expr: CasADi expressions for constraints (or None for no constraints)
2. lower_bounds: CasADi expressions for constraint lower bounds
3. upper_bounds: CasADi expressions for constraint upper bounds

AVAILABLE STATE VARIABLES:
- x_k[0:3]: COM position [x, y, z]
- x_k[3:6]: COM velocity [vx, vy, vz]
- x_k[6:9]: Euler angles [roll, pitch, yaw]
- x_k[9:12]: Angular velocity [wx, wy, wz]
- x_k[12:24]: Joint angles (12 joints: 4 legs × 3 joints each)

AVAILABLE INPUT VARIABLES:
- u_k[0:12]: Joint velocities (12 joints)
- u_k[12:24]: Ground reaction forces (4 feet × 3 forces each: fx, fy, fz)

KINODYNAMIC MODEL FUNCTIONS:
- kindyn_model.forward_kinematics_FL_fun(H, joint_pos) -> 4x4 transformation matrix for front left foot
- kindyn_model.forward_kinematics_FR_fun(H, joint_pos) -> front right foot
- kindyn_model.forward_kinematics_RL_fun(H, joint_pos) -> rear left foot
- kindyn_model.forward_kinematics_RR_fun(H, joint_pos) -> rear right foot
- kindyn_model.jacobian_FL_fun(H, joint_pos) -> Jacobian for front left foot
- (similar for FR, RL, RR)

Where H is the homogeneous transformation matrix:
```python
w_R_b = SO3.from_euler(vertcat(x_k[6], x_k[7], x_k[8])).as_matrix()
H = MX.eye(4)
H[0:3, 0:3] = w_R_b
H[0:3, 3] = x_k[0:3]  # COM position
```

CONFIG PARAMETERS:
- config.mpc_config.mpc_dt: Time step
- config.experiment.mu_ground: Ground friction coefficient
- config.robot_data.mass: Robot mass
- config.robot_data.grf_limits: Max ground reaction force

CONSTRAINT EXAMPLES:

1. Terminal pitch constraint (backflip):
```python
def task_specific_constraints(x_k, u_k, kindyn_model, config, contact_k):
    # Enforce 2π rotation in pitch for backflip
    pitch_constraint = x_k[7]  # pitch angle
    lower_bound = 2 * 3.14159 - 0.1  # Nearly 2π rotation
    upper_bound = 2 * 3.14159 + 0.1
    return pitch_constraint, lower_bound, upper_bound
```

2. Jump height constraint:
```python
def task_specific_constraints(x_k, u_k, kindyn_model, config, contact_k):
    # COM must reach minimum height during flight
    com_height = x_k[2]
    min_height = 0.5  # 50cm jump
    return com_height, min_height, inf
```

3. Body orientation constraint:
```python
def task_specific_constraints(x_k, u_k, kindyn_model, config, contact_k):
    # Keep body level during motion
    roll, pitch = x_k[6], x_k[7]
    orientation_constraints = vertcat(roll, pitch)
    bounds_lower = vertcat(-0.2, -0.2)  # ±11 degrees
    bounds_upper = vertcat(0.2, 0.2)
    return orientation_constraints, bounds_lower, bounds_upper
```

IMPORTANT GUIDELINES:
1. Output ONLY Python code - no markdown code blocks, no explanations
2. All required modules are available: cs, np, math, SO3, MX, vertcat, inf, etc.
3. NEVER import modules - everything is already available in the namespace
4. Use vertcat() to stack multiple constraints
5. Use inf for unbounded constraints
6. Consider the robot's physical limitations
7. Constraints apply at each time step of the trajectory
8. Use contact_k to check foot contact status (contact_k[i] = 1 means foot i is in contact)
9. Common functions available: sin, cos, sqrt, fabs, fmax, fmin, sum1, eye
10. When given error feedback, fix the issues and regenerate the complete code

EXAMPLES OF VALID STRUCTURES:

Simple function:
```
def my_constraints(x_k, u_k, kindyn_model, config, contact_k):
    height = x_k[2]
    return height, 0.2, inf
```

Multiple helper functions:
```
def compute_foot_positions(x_k, kindyn_model):
    # ... helper logic
    return foot_pos

def main_constraint_func(x_k, u_k, kindyn_model, config, contact_k):
    positions = compute_foot_positions(x_k, kindyn_model)
    # ... main logic
    return constraints, lower, upper
```

Generate code that achieves the user's desired behavior while respecting physics and robot limitations."""

    def get_user_prompt(self, command: str) -> str:
        """
        Create the initial user prompt from a natural language command.

        Args:
            command: Natural language command (e.g., "do a backflip")

        Returns:
            Formatted user prompt
        """
        return f"""TASK: {command}

Generate Python code that implements constraints to enable the quadruped robot to perform: "{command}"

Consider:
1. What are the key kinematic requirements for this behavior?
2. What constraints on body orientation, position, or timing are needed?
3. How should the robot's state evolve over the trajectory?
4. Are there safety constraints to prevent self-collision or ground penetration?

Generate ONLY the Python code - no explanations, no markdown."""

    def create_feedback_context(
        self,
        iteration: int,
        trajectory_data: Dict[str, Any],
        optimization_status: Dict[str, Any],
        simulation_results: Dict[str, Any],
        previous_constraints: str,
    ) -> str:
        """
        Create feedback context for the next LLM iteration.

        Args:
            iteration: Current iteration number
            trajectory_data: Optimized trajectory information
            optimization_status: Solver status and convergence info
            simulation_results: Simulation execution results
            previous_constraints: Previously generated constraints

        Returns:
            Formatted feedback context string
        """
        # Build context safely to avoid f-string formatting issues
        context_parts = [
            f"ITERATION {iteration} FEEDBACK:",
            "",
            "PREVIOUS CONSTRAINTS:",
            str(previous_constraints),
            "",
            "OPTIMIZATION RESULTS:",
            f"- Status: {optimization_status.get('status', 'unknown')}",
            f"- Converged: {optimization_status.get('converged', False)}",
            f"- Objective Value: {optimization_status.get('objective_value', 'N/A')}",
            f"- Constraint Violations: {optimization_status.get('constraint_violations', 'N/A')}",
            "",
            "TRAJECTORY ANALYSIS:",
        ]

        # Add trajectory analysis safely
        max_height = trajectory_data.get("max_com_height", "N/A")
        if isinstance(max_height, (int, float)):
            context_parts.append(f"- Max COM Height: {max_height:.3f}m")
        else:
            context_parts.append(f"- Max COM Height: {max_height}")

        final_pitch = trajectory_data.get("final_pitch", "N/A")
        if isinstance(final_pitch, (int, float)):
            context_parts.append(f"- Final Pitch Angle: {final_pitch:.3f}rad")
        else:
            context_parts.append(f"- Final Pitch Angle: {final_pitch}")

        total_rotation = trajectory_data.get("total_pitch_rotation", "N/A")
        if isinstance(total_rotation, (int, float)):
            context_parts.append(f"- Total Rotation: {total_rotation:.3f}rad")
        else:
            context_parts.append(f"- Total Rotation: {total_rotation}")

        flight_duration = trajectory_data.get("flight_duration", "N/A")
        if isinstance(flight_duration, (int, float)):
            context_parts.append(f"- Flight Duration: {flight_duration:.3f}s")
        else:
            context_parts.append(f"- Flight Duration: {flight_duration}")

        max_angular_vel = trajectory_data.get("max_angular_vel", "N/A")
        if isinstance(max_angular_vel, (int, float)):
            context_parts.append(f"- Max Angular Velocity: {max_angular_vel:.3f}rad/s")
        else:
            context_parts.append(f"- Max Angular Velocity: {max_angular_vel}")

        context_parts.extend(
            [
                "",
                "SIMULATION RESULTS:",
                f"- Execution Success: {simulation_results.get('success', False)}",
                f"- Tracking Error: {simulation_results.get('tracking_error', 'N/A')}",
                f"- Ground Contacts: {simulation_results.get('unexpected_contacts', 'N/A')}",
                f"- Physical Realism: {simulation_results.get('realistic', 'Unknown')}",
                "",
                "ISSUES IDENTIFIED:",
            ]
        )

        context = "\n".join(context_parts) + "\n"

        # Add specific issues based on results
        issues = []

        if not optimization_status.get("converged", False):
            issues.append(
                "- Optimization failed to converge - constraints may be too strict or conflicting"
            )

        if trajectory_data.get("total_pitch_rotation", 0) < 5.5:  # Less than ~2π
            issues.append("- Insufficient rotation for complete backflip")

        if trajectory_data.get("max_com_height", 0) < 0.3:
            issues.append(
                "- Robot may not achieve sufficient height for aerial maneuver"
            )

        if simulation_results.get("tracking_error", float("inf")) > 0.1:
            issues.append(
                "- Large tracking error between planned and executed trajectory"
            )

        if not simulation_results.get("success", False):
            issues.append("- Simulation execution failed or was unrealistic")

        if not issues:
            issues.append("- No major issues identified")

        context += "\\n".join(issues)

        context += """

REFINEMENT REQUEST:
Based on the analysis above, please modify the constraints to address the identified issues.
Focus on:
1. Ensuring optimization convergence
2. Achieving the desired kinematic behavior
3. Maintaining physical realism
4. Improving trajectory tracking

Provide the updated task_specific_constraints function."""

        # Store iteration history
        self.iteration_history.append(
            {
                "iteration": iteration,
                "trajectory_data": trajectory_data,
                "optimization_status": optimization_status,
                "simulation_results": simulation_results,
                "constraints": previous_constraints,
            }
        )

        return context

    def create_repair_prompt(
        self, command: str, failed_code: str, error_message: str, attempt_number: int
    ) -> str:
        """
        Create a prompt to ask the LLM to fix failed constraint code.

        Args:
            command: Original natural language command
            failed_code: The code that failed
            error_message: Error message from SafeExecutor/MPC
            attempt_number: Which attempt this is (1-10)

        Returns:
            Repair prompt string
        """
        # Truncate code if too long
        code_snippet = failed_code
        if len(failed_code) > 1000:
            code_snippet = (
                failed_code[:500] + "\n... [truncated] ...\n" + failed_code[-500:]
            )

        return f"""TASK: {command}

ATTEMPT {attempt_number}/10 - CODE REPAIR NEEDED

Your previous code failed with this error:
```
{error_message}
```

Failed code:
```python
{code_snippet}
```

Fix the error and regenerate the complete Python code for: "{command}"

Common issues to check:
1. Function signature must accept exactly 5 arguments: (x_k, u_k, kindyn_model, config, contact_k)
2. Must return tuple of exactly 3 elements: (constraint_expr, lower_bounds, upper_bounds)
3. Use available functions directly: vertcat, inf, sin, cos, SO3, MX, eye
4. No imports allowed - all modules already available
5. CasADi expressions must be properly formed
6. Check for syntax errors, undefined variables, wrong argument counts

Generate ONLY the corrected Python code - no explanations, no markdown."""

    def analyze_trajectory(
        self, state_traj: np.ndarray, mpc_dt: float
    ) -> Dict[str, Any]:
        """
        Analyze trajectory to extract key metrics for feedback.

        Args:
            state_traj: State trajectory array (N x 24)
            mpc_dt: Time step

        Returns:
            Dictionary of trajectory metrics
        """
        if state_traj.shape[0] == 0:
            return {}

        # Extract key state components
        com_positions = state_traj[:, 0:3]  # x, y, z
        com_velocities = state_traj[:, 3:6]
        euler_angles = state_traj[:, 6:9]  # roll, pitch, yaw
        angular_velocities = state_traj[:, 9:12]

        # Calculate metrics
        metrics = {
            "max_com_height": np.max(com_positions[:, 2]),
            "min_com_height": np.min(com_positions[:, 2]),
            "final_pitch": euler_angles[-1, 1],
            "initial_pitch": euler_angles[0, 1],
            "total_pitch_rotation": euler_angles[-1, 1] - euler_angles[0, 1],
            "max_angular_vel": np.max(np.linalg.norm(angular_velocities, axis=1)),
            "trajectory_duration": len(state_traj) * mpc_dt,
            "max_com_velocity": np.max(np.linalg.norm(com_velocities, axis=1)),
        }

        # Estimate flight duration (when COM z-velocity > 0 and height > initial height)
        initial_height = com_positions[0, 2]
        airborne_mask = (com_positions[:, 2] > initial_height + 0.05) & (
            com_velocities[:, 2] > -0.1
        )
        flight_duration = np.sum(airborne_mask) * mpc_dt
        metrics["flight_duration"] = flight_duration

        return metrics
