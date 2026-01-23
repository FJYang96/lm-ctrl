"""LLM prompts for constraint generation and feedback."""

from typing import Any


def get_system_prompt(mass: float = 15.0, initial_height: float = 0.2117) -> str:
    """
    Get the system prompt that instructs the LLM on MPC configuration + constraint generation.

    Args:
        mass: Robot mass in kg (from actual config)
        initial_height: Robot initial COM height in meters (from actual config)

    Returns:
        System prompt string with accurate physical parameters
    """
    return f"""You are a robotics expert generating MPC configurations for quadruped robot trajectory optimization.

OUTPUT FORMAT: Return ONLY Python code. You may use ```python code blocks.

== CRITICAL: MANDATORY REQUIREMENTS ==

YOUR CODE WILL FAIL WITHOUT ALL FIVE OF THESE CALLS:

1. mpc.set_task_name("...")           <- REQUIRED
2. mpc.set_duration(...)              <- REQUIRED
3. mpc.set_time_step(0.02)            <- REQUIRED
4. mpc.set_contact_sequence(...)      <- REQUIRED - #1 CAUSE OF FAILURES
5. mpc.add_constraint(...)            <- REQUIRED

THE MOST COMMON FAILURE: Forgetting mpc.set_contact_sequence()
- EVERY motion needs a contact sequence, including ground-based motions like squatting
- For ground-based motions: use [1,1,1,1] for all timesteps (all feet grounded)
- For aerial motions: include [0,0,0,0] phases (all feet in air)
- Use mpc._create_phase_sequence() to build the contact array

If you see "No contact sequence specified" error, you forgot mpc.set_contact_sequence().

== ROBOT PHYSICS ==

State x_k (24-dim):
- x_k[0:3]: COM position [x, y, z] in meters
- x_k[3:6]: COM velocity [vx, vy, vz] in m/s
- x_k[6:9]: orientation [roll, pitch, yaw] in radians
- x_k[9:12]: angular velocity [wx, wy, wz] in rad/s
- x_k[12:24]: joint angles (12 joints)

Key physical facts (from actual robot config):
- Robot mass: {mass:.2f} kg
- Robot starts at COM height EXACTLY {initial_height:.4f}m
- Achievable jump height: ~0.3-0.5m above starting height
- Rotation: angle_change = angular_velocity * time
- Projectile motion: peak_height = initial_height + v^2/(2g)

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
    return (constraint_expr, lower_bound, upper_bound)  # CasADi MX expressions

PRINCIPLE 1: NEVER violate constraints at t=0
- At k=0, the robot is at its starting state (height={initial_height:.4f}m)
- If you set upper_bound < {initial_height:.4f} at k=0, optimization FAILS IMMEDIATELY
- Always use bounds that INCLUDE the starting state at the first timestep
- Then gradually tighten constraints over time using progress = k/horizon

PRINCIPLE 2: Constraints define FEASIBLE REGIONS, not exact trajectories
- The optimizer finds the EASIEST path within your constraints
- If starting state is already valid, optimizer may do nothing
- To FORCE motion: make constraints that EXCLUDE the starting state AFTER t=0

PRINCIPLE 3: Start conservative, fail fast
- Loose constraints -> optimization succeeds -> check if motion happened
- Tight constraints -> optimization fails -> you learn nothing
- Better to succeed with weak motion than fail completely

PRINCIPLE 4: One-sided bounds are more robust
- Use (lower, cs.inf) or (-cs.inf, upper) when possible
- Tight bands (lower, upper) often cause infeasibility
- Let the optimizer find the optimal value within your region

PRINCIPLE 5: Time-varying constraints guide motion
- Use progress = k/horizon for gradual changes
- Example for lowering: if k == 0: upper = {initial_height:.4f} + 0.01 else: upper = {initial_height:.4f} - progress * drop
- Smooth ramps are more feasible than sudden jumps

== AVAILABLE FUNCTIONS ==
vertcat, horzcat, mtimes, sin, cos, tan, sqrt, exp, log, fabs, fmax, fmin,
sum1, norm_2, atan2, asin, acos, tanh, MX, DM, cs.inf, pi, np

== EXAMPLE STRUCTURE ==

mpc.set_task_name("task")
mpc.set_duration(1.5)
mpc.set_time_step(0.02)
phases = [("phase1", 0.3, [1,1,1,1]), ("phase2", 0.9, [0,0,0,0]), ("phase3", 0.3, [1,1,1,1])]
contact_seq = mpc._create_phase_sequence(phases)
mpc.set_contact_sequence(contact_seq)

def task_constraints(x_k, u_k, kindyn_model, config, contact_k, k, horizon):
    progress = k / horizon
    height = x_k[2]

    # IMPORTANT: Don't violate at t=0! Start with loose bounds.
    if k == 0:
        lower = 0.05
        upper = 0.5  # Above starting height - always feasible
    else:
        # Now gradually tighten to force motion
        lower = 0.05
        upper = {initial_height:.4f} + 0.01 - progress * 0.1  # Gradually lower ceiling

    return (height, lower, upper)

mpc.add_constraint(task_constraints)

== UNDERSTANDING FEEDBACK ==

You will receive rich feedback after each iteration:

1. VISUAL FRAMES: Comparing PLANNED vs SIMULATED trajectory frames.

2. TASK PROGRESS TABLE: Shows % completion toward each goal criterion.
   - If a criterion is at 25%, your constraint for that is too weak.
   - If a criterion is at 100%, that aspect is working - focus on others.

3. PHASE ANALYSIS: Breakdown by motion phase (stance, flight, landing).

4. GROUND REACTION FORCES: Takeoff GRF should be 2-3x body weight for good jumps.

5. ACTUATOR STATUS: Shows if robot is at physical limits.

6. VS PREVIOUS ITERATION: Shows what improved or regressed.

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
2. What contact sequence is appropriate? (REQUIRED - your code will fail without mpc.set_contact_sequence())
3. What physical quantities need to be constrained to achieve this?
4. What are reasonable bounds that FORCE the desired motion?

REMINDER: You MUST include mpc.set_contact_sequence() - this is the #1 cause of failures.

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
        # Get rotation values
        pitch_rotation = trajectory_data.get("total_pitch_rotation", 0)
        yaw_rotation = trajectory_data.get("max_yaw", 0)  # Total yaw change
        roll_rotation = trajectory_data.get("max_roll", 0)

        context += f"""
ACHIEVED TRAJECTORY:
- Height: {trajectory_data.get('initial_com_height', 0):.3f}m -> max {trajectory_data.get('max_com_height', 0):.3f}m -> final {trajectory_data.get('final_com_height', 0):.3f}m
- Height change: {trajectory_data.get('height_gain', 0):.3f}m
- Pitch rotation (forward/back flip): {pitch_rotation:.2f} rad ({pitch_rotation * 57.3:.0f} deg)
- Yaw rotation (turning left/right): {yaw_rotation:.2f} rad ({yaw_rotation * 57.3:.0f} deg)
- Roll rotation (side tilt): {roll_rotation:.2f} rad ({roll_rotation * 57.3:.0f} deg)
- Flight duration: {trajectory_data.get('flight_duration', 0):.2f}s
"""

        # Analyze what might be missing
        height_gain = trajectory_data.get("height_gain", 0)
        pitch_rot = abs(pitch_rotation)
        yaw_rot = abs(yaw_rotation)

        suggestions = []

        if height_gain < 0.05 and height_gain > -0.05:
            suggestions.append(
                "Height barely changed. To force vertical motion, constrain height after t=0 to exclude starting value."
            )

        if pitch_rot < 0.5:  # Less than ~30 degrees
            suggestions.append(
                "Minimal pitch rotation. To force pitch (backflip/frontflip), constrain x_k[10] (pitch angular velocity) to be non-zero during flight."
            )

        if yaw_rot < 0.5:  # Less than ~30 degrees
            suggestions.append(
                "Minimal yaw rotation. To force turning, constrain x_k[11] (yaw angular velocity) or x_k[8] (yaw angle) to change."
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

⚠️ MANDATORY CHECKLIST - Your code MUST include ALL of these:
□ mpc.set_task_name("...")
□ mpc.set_duration(...)
□ mpc.set_time_step(0.02)
□ mpc.set_contact_sequence(...)  ← THIS IS THE #1 MISSING CALL
□ mpc.add_constraint(...)

Other requirements:
- Constraint function must have 7 parameters: (x_k, u_k, kindyn_model, config, contact_k, k, horizon)
- Must return exactly 3 values: (constraint_expr, lower_bound, upper_bound)
- All return values must be CasADi MX expressions (use vertcat for multiple constraints)
- CRITICAL: At k=0, bounds must INCLUDE the starting state (height=0.2117m). Constraints that violate t=0 cause immediate failure!

Return ONLY corrected Python code."""
