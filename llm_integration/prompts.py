"""LLM prompts for constraint generation and feedback."""

from typing import Any


def get_system_prompt() -> str:
    """
    Get the system prompt that instructs the LLM on MPC configuration + constraint generation.

    Returns:
        System prompt string
    """
    return """You are a robotics expert generating COMPLETE MPC configurations for quadruped robot behaviors.

CRITICAL REQUIREMENTS - FOLLOW EXACTLY:

1. OUTPUT FORMAT: Return ONLY Python code. NO markdown, NO backticks, NO explanations, NO comments.

2. GENERATE TWO PARTS:
   A) MPC CONFIGURATION CODE - Sets up contact sequences and timing for the task
   B) CONSTRAINT FUNCTION - Enforces task-specific behavior requirements

3. MPC CONFIGURATION: Use the 'mpc' object to configure the task:
   - mpc.set_task_name("task_name")
   - mpc.set_duration(total_seconds)
   - mpc.set_time_step(0.02)  # Usually keep at 0.02
   - mpc.set_contact_sequence(contact_array)
   - mpc.add_constraint(constraint_function)

4. CONTACT SEQUENCES: Choose appropriate foot contact patterns:
   - [1,1,1,1] = all feet in contact (for turns, squats, walking)
   - [0,0,0,0] = flight phase (for jumps, leaps, flips)
   - Mixed patterns for complex motions

5. CONSTRAINT SIGNATURE: def name(x_k, u_k, kindyn_model, config, contact_k):
   MUST return exactly 3 CasADi MX objects: constraint_expr, lower_bounds, upper_bounds

ROBOT STATE SPACE (24-DOF):
x_k[0:3]   - COM position [x, y, z]
x_k[3:6]   - COM velocity [vx, vy, vz]
x_k[6]     - roll angle
x_k[7]     - pitch angle
x_k[8]     - yaw angle
x_k[9:12]  - angular velocity [wx, wy, wz]
x_k[12:24] - joint angles (12 joints: 4 legs × 3 joints)

ROBOT INPUT SPACE (24-DOF):
u_k[0:12]  - joint velocities (12 joints)
u_k[12:24] - ground reaction forces (4 feet × 3 forces: fx,fy,fz per foot)

AVAILABLE FUNCTIONS:
vertcat, horzcat, sin, cos, sqrt, fabs, fmax, fmin, sum1, MX, inf, pi, np

CONTACT SEQUENCE HELPERS:
- np.ones((4, horizon)) - all feet in contact
- np.zeros((4, horizon)) - all feet in flight
- mpc._create_phase_sequence([(phase_name, duration, [FL,FR,RL,RR]), ...])

TASK-SPECIFIC MPC EXAMPLES:

BACKFLIP (needs flight phase + rotation):
mpc.set_task_name("backflip")
mpc.set_duration(1.2)
phases = [("stance", 0.3, [1,1,1,1]), ("flight", 0.6, [0,0,0,0]), ("landing", 0.3, [1,1,1,1])]
contact_seq = mpc._create_phase_sequence(phases)
mpc.set_contact_sequence(contact_seq)

def backflip_constraints(x_k, u_k, kindyn_model, config, contact_k):
    pitch = x_k[7]
    height = x_k[2]
    constraints = vertcat(pitch, height)
    lower = vertcat(2*pi - 0.2, 0.4)
    upper = vertcat(2*pi + 0.2, inf)
    return constraints, lower, upper
mpc.add_constraint(backflip_constraints)

TURN AROUND (stay grounded + rotate):
mpc.set_task_name("turn_around")
mpc.set_duration(2.0)
contact_seq = np.ones((4, int(2.0/0.02)))
mpc.set_contact_sequence(contact_seq)

def turn_constraints(x_k, u_k, kindyn_model, config, contact_k):
    yaw = x_k[8]
    height = x_k[2]
    constraints = vertcat(yaw, height)
    lower = vertcat(pi - 0.1, 0.15)
    upper = vertcat(pi + 0.1, 0.35)
    return constraints, lower, upper
mpc.add_constraint(turn_constraints)

JUMP LEFT (takeoff + lateral displacement):
mpc.set_task_name("jump_left")
mpc.set_duration(1.0)
phases = [("prep", 0.2, [1,1,1,1]), ("flight", 0.6, [0,0,0,0]), ("land", 0.2, [1,1,1,1])]
contact_seq = mpc._create_phase_sequence(phases)
mpc.set_contact_sequence(contact_seq)

def jump_left_constraints(x_k, u_k, kindyn_model, config, contact_k):
    com_y = x_k[1]
    height = x_k[2]
    constraints = vertcat(com_y, height)
    lower = vertcat(0.3, 0.3)
    upper = vertcat(inf, inf)
    return constraints, lower, upper
mpc.add_constraint(jump_left_constraints)

SQUAT DOWN (stay grounded + lower):
mpc.set_task_name("squat")
mpc.set_duration(1.5)
contact_seq = np.ones((4, int(1.5/0.02)))
mpc.set_contact_sequence(contact_seq)

def squat_constraints(x_k, u_k, kindyn_model, config, contact_k):
    height = x_k[2]
    return height, 0.1, 0.2
mpc.add_constraint(squat_constraints)

CRITICAL RULES:
- ALWAYS configure MPC first, then define constraints
- ALWAYS use appropriate contact sequences for the task type
- NEVER use import statements
- ALWAYS return 3 CasADi MX objects from constraint functions
- CHOOSE realistic durations (0.5-3.0 seconds typical)
- KEEP feet planted (all 1s) for turning/ground-based motions
- USE flight phases (0s) only for jumps/flips/leaps

YOUR TASK: Generate complete MPC configuration + constraints for the given robot behavior."""


def get_user_prompt(command: str) -> str:
    """
    Create the initial user prompt from a natural language command.

    Args:
        command: Natural language command (e.g., "do a backflip")

    Returns:
        Formatted user prompt
    """
    return f"""TASK: {command}

Generate complete MPC configuration and constraints for: "{command}"

REQUIREMENTS:
- Return ONLY Python code (no markdown, no explanations)
- Configure MPC object with appropriate contact sequence and timing
- Define constraint function with correct signature
- Choose contact patterns suitable for the task type

STRUCTURE YOUR CODE:
1. Set task name, duration, and time step
2. Create appropriate contact sequence for the behavior
3. Define constraint function that enforces the behavior
4. Add constraint to MPC

BEHAVIOR ANALYSIS:
- Ground-based (turn, walk, squat): keep feet planted [1,1,1,1]
- Aerial (jump, flip, leap): include flight phase [0,0,0,0]
- Mixed motions: use phase sequences

Generate complete MPC setup for "{command}":"""


def create_feedback_context(
    iteration: int,
    trajectory_data: dict[str, Any],
    optimization_status: dict[str, Any],
    simulation_results: dict[str, Any],
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
    # Build simplified feedback context
    context = f"""ITERATION {iteration} FEEDBACK - IMPROVE CONSTRAINTS

PREVIOUS CODE:
{previous_constraints}

RESULTS ANALYSIS:
Optimization: {"SUCCESS" if optimization_status.get("converged", False) else "FAILED"}
Simulation: {"SUCCESS" if simulation_results.get("success", False) else "FAILED"}"""

    # Add specific trajectory metrics
    if trajectory_data:
        max_height = trajectory_data.get("max_com_height", 0)
        total_rotation = trajectory_data.get("total_pitch_rotation", 0)
        flight_duration = trajectory_data.get("flight_duration", 0)

        context += f"""
Max Height: {max_height:.3f}m
Total Rotation: {total_rotation:.3f}rad ({total_rotation * 57.3:.1f}°)
Flight Duration: {flight_duration:.3f}s"""

    # Add specific improvement suggestions
    issues = []
    if not optimization_status.get("converged", False):
        issues.append("Optimization failed - constraints too strict or conflicting")

    if simulation_results.get("tracking_error", float("inf")) > 0.1:
        issues.append("High tracking error - constraints may be unrealistic")

    if not simulation_results.get("realistic", True):
        issues.append("Simulation unrealistic - constraints cause physical violations")

    if issues:
        context += "\n\nPROBLEMS:\n" + "\n".join(f"- {issue}" for issue in issues)

    context += """

TASK: Generate improved constraint function.
- Return ONLY Python code (no explanations)
- Fix identified issues
- Keep same function signature
- Ensure bounds match constraint dimensions
- Use CasADi MX objects only"""

    return context


def create_repair_prompt(
    command: str, failed_code: str, error_message: str, attempt_number: int
) -> str:
    """
    Create a prompt to ask the LLM to fix failed MPC configuration code.

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
    if len(failed_code) > 800:
        code_snippet = (
            failed_code[:400] + "\n... [truncated] ...\n" + failed_code[-400:]
        )

    return f"""MPC CONFIGURATION REPAIR ATTEMPT {attempt_number}/10

TASK: {command}

ERROR MESSAGE:
{error_message}

FAILED CODE:
{code_snippet}

FIX THE ERROR AND REGENERATE COMPLETE MPC CONFIGURATION

COMMON FIXES:
- Missing mpc configuration: must call mpc.set_contact_sequence() and mpc.add_constraint()
- Wrong constraint signature: must be (x_k, u_k, kindyn_model, config, contact_k)
- Wrong return format: constraints must return exactly 3 CasADi MX objects
- Contact sequence errors: use np.ones() or mpc._create_phase_sequence()
- Dimension mismatches: bounds must match constraint vector length
- Import errors: remove all import statements
- Syntax errors: check parentheses, brackets, indentation

REQUIREMENTS:
- Return ONLY Python code (no explanations, no markdown)
- Configure MPC: set_task_name, set_duration, set_contact_sequence
- Define constraint function with correct signature
- Add constraint to MPC with mpc.add_constraint()
- Use appropriate contact sequence for the task type

Generate corrected MPC configuration:"""
