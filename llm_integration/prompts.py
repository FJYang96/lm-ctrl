"""LLM prompts for constraint generation and feedback."""

from typing import Any


def get_system_prompt() -> str:
    """
    Get the system prompt that instructs the LLM on MPC configuration + constraint generation.

    Returns:
        System prompt string
    """
    return """You are a robotics expert generating MPC configurations for quadruped robot trajectory optimization.

OUTPUT FORMAT: Return ONLY Python code. No markdown, no backticks, no explanations.

== ROBOT PHYSICS ==

State x_k (24-dim):
- x_k[0:3]: COM position [x, y, z] in meters
- x_k[3:6]: COM velocity [vx, vy, vz] in m/s
- x_k[6:9]: orientation [roll, pitch, yaw] in radians
- x_k[9:12]: angular velocity [wx, wy, wz] in rad/s
- x_k[12:24]: joint angles (12 joints)

Key physical facts:
- Robot starts at COM height ~0.21m
- Mass ~12kg, can jump to ~0.5-1.5m height
- Rotation: angle_change ≈ angular_velocity × time
- Projectile motion: peak_height ≈ initial_height + v²/(2g)

== MPC CONFIGURATION ==

Required calls:
  mpc.set_task_name("name")
  mpc.set_duration(seconds)  # typically 1.0-2.0s
  mpc.set_time_step(0.02)
  mpc.set_contact_sequence(contact_array)
  mpc.add_constraint(constraint_function)

Contact patterns:
- [1,1,1,1] = all feet grounded (walking, turning, squatting)
- [0,0,0,0] = flight phase (jumping, flipping)
- Use mpc._create_phase_sequence([(name, duration, pattern), ...])

== CONSTRAINT DESIGN PRINCIPLES ==

Signature: def name(x_k, u_k, kindyn_model, config, contact_k, k, horizon):
    # k = current timestep, horizon = total timesteps
    # progress = k / horizon  (0.0 at start, 1.0 at end)
    return (constraint_expr, lower_bound, upper_bound)  # CasADi MX objects

PRINCIPLE 1: Constraints define FEASIBLE REGIONS, not exact trajectories
- The optimizer finds the EASIEST path within your constraints
- If starting state is already valid, optimizer may do nothing
- To FORCE motion: make constraints that EXCLUDE the starting state

PRINCIPLE 2: Start conservative, fail fast
- Loose constraints → optimization succeeds → check if motion happened
- Tight constraints → optimization fails → you learn nothing
- Better to succeed with weak motion than fail completely

PRINCIPLE 3: One-sided bounds are more robust
- (lower, inf) or (-inf, upper) usually works
- (lower, upper) with tight band often fails
- Let the optimizer find the optimal value within your region

PRINCIPLE 4: Time-varying constraints guide motion
- Use progress = k/horizon for gradual changes
- Example: upper_bound = start_value - progress * change
- Smooth ramps are more feasible than sudden jumps

PRINCIPLE 5: Understand what you're actually constraining
- height >= 0.5 means "never go below 0.5m" at ANY timestep
- This includes takeoff! Robot starts at 0.21m, can't instantly be at 0.5m
- Think about the ENTIRE trajectory, not just the goal state

== AVAILABLE FUNCTIONS ==
vertcat, horzcat, sin, cos, sqrt, fabs, fmax, fmin, sum1, MX, inf, pi, np

== EXAMPLE STRUCTURE ==

mpc.set_task_name("task")
mpc.set_duration(1.5)
mpc.set_time_step(0.02)
phases = [("phase1", 0.3, [1,1,1,1]), ("phase2", 0.9, [0,0,0,0]), ("phase3", 0.3, [1,1,1,1])]
contact_seq = mpc._create_phase_sequence(phases)
mpc.set_contact_sequence(contact_seq)

def task_constraints(x_k, u_k, kindyn_model, config, contact_k, k, horizon):
    progress = k / horizon
    # Extract relevant states
    # Apply phase-appropriate constraints
    # Return (expr, lower, upper)
    return constraints, lower, upper

mpc.add_constraint(task_constraints)

== TASK ==
Generate MPC configuration and constraints for the requested behavior.
Think about: What motion is needed? What constraints will FORCE that motion?
Start with simple, loose constraints. The feedback loop will help you refine."""


def get_user_prompt(command: str) -> str:
    """
    Create the initial user prompt from a natural language command.

    Args:
        command: Natural language command (e.g., "do a backflip")

    Returns:
        Formatted user prompt
    """
    return f"""Generate MPC configuration for: "{command}"

Think step by step:
1. What type of motion is this? (ground-based, aerial, rotation, translation)
2. What contact sequence is appropriate?
3. What physical quantities need to be constrained to achieve this?
4. What are reasonable bounds that FORCE the desired motion?

Return ONLY Python code."""


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
    context = f"""ITERATION {iteration} FEEDBACK

PREVIOUS CODE:
{previous_constraints}

"""

    # Optimization status
    if optimization_status.get("converged", False):
        context += "OPTIMIZATION: SUCCESS\n"
    else:
        context += """OPTIMIZATION: FAILED - No feasible trajectory found!

This means your constraints are MUTUALLY EXCLUSIVE or PHYSICALLY IMPOSSIBLE.
Common causes:
- Requiring height > X when robot starts below X (can't teleport)
- Combining too many tight constraints simultaneously
- Progressive bounds that increase faster than physics allows

To fix: SIMPLIFY and LOOSEN constraints. Remove the most restrictive one.
Start with just ONE key constraint, verify it works, then add more gradually.

"""

    # Simulation status
    if simulation_results.get("success", False):
        context += f"SIMULATION: SUCCESS (tracking_error: {simulation_results.get('tracking_error', 0):.3f})\n"
    else:
        context += "SIMULATION: FAILED\n"

    # Trajectory metrics
    if trajectory_data and optimization_status.get("converged", False):
        context += f"""
ACHIEVED TRAJECTORY:
- Height: {trajectory_data.get('initial_com_height', 0):.3f}m → max {trajectory_data.get('max_com_height', 0):.3f}m → final {trajectory_data.get('final_com_height', 0):.3f}m
- Height change: {trajectory_data.get('height_gain', 0):.3f}m
- Pitch rotation: {trajectory_data.get('total_pitch_rotation', 0):.2f} rad ({trajectory_data.get('total_pitch_rotation', 0) * 57.3:.0f}°)
- Max pitch: {trajectory_data.get('max_pitch', 0):.2f} rad ({trajectory_data.get('max_pitch', 0) * 57.3:.0f}°)
- Flight duration: {trajectory_data.get('flight_duration', 0):.2f}s
"""

        # Analyze what might be missing
        height_gain = trajectory_data.get("height_gain", 0)
        rotation = abs(trajectory_data.get("total_pitch_rotation", 0))

        suggestions = []

        if height_gain < 0.05 and height_gain > -0.05:
            suggestions.append(
                "Height barely changed. To force vertical motion, constrain height to exclude starting value (0.21m)."
            )

        if rotation < 0.5:  # Less than ~30 degrees
            suggestions.append(
                "Minimal rotation occurred. To force rotation, constrain angular velocity (wy for pitch, wz for yaw) to be non-zero."
            )

        if suggestions:
            context += "\nANALYSIS:\n" + "\n".join(f"- {s}" for s in suggestions)

    context += """

TASK: Generate improved constraints based on this feedback.
Return ONLY Python code."""

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
    code_snippet = failed_code
    if len(failed_code) > 800:
        code_snippet = (
            failed_code[:400] + "\n... [truncated] ...\n" + failed_code[-400:]
        )

    return f"""REPAIR ATTEMPT {attempt_number}/10

TASK: {command}

ERROR: {error_message}

FAILED CODE:
{code_snippet}

Common fixes:
- Constraint function must have 7 parameters: (x_k, u_k, kindyn_model, config, contact_k, k, horizon)
- Must return exactly 3 values: (constraint_expr, lower_bound, upper_bound)
- All return values must be CasADi MX objects (use vertcat for multiple constraints)
- No import statements allowed
- Must call mpc.set_contact_sequence() and mpc.add_constraint()

Return ONLY corrected Python code."""
