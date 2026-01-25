"""LLM prompts for constraint generation and feedback."""


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

PRINCIPLE 1: Bounds must be CONTINUOUS across timesteps
- The optimizer needs a SMOOTH path from start to goal
- NEVER use if/else branches that create sudden jumps in bounds
- If lower_bound at timestep k+1 > upper_bound at timestep k, optimization FAILS

PRINCIPLE 2: Start from the initial state
- Robot starts at height={initial_height:.4f}m, all angles=0, all velocities=0
- At k=0, bounds MUST include these values or optimization fails immediately
- Use formulas that evaluate to valid bounds at progress=0

PRINCIPLE 3: Use SMOOTH RAMPS, not step functions
- Express bounds as linear functions of progress: bound = start_value + progress * change
- The optimizer will find the optimal trajectory within these smooth bounds

PRINCIPLE 4: One-sided bounds are more robust
- Use (lower, cs.inf) or (-cs.inf, upper) when possible
- Only constrain what you NEED - don't over-constrain

PRINCIPLE 5: Constrain the GOAL, not the path
- Don't try to script the entire trajectory with tight bounds at every timestep
- Instead: loose bounds throughout, tight bounds only at the END

PRINCIPLE 6: Constraints define FEASIBLE REGIONS, not exact trajectories
- The optimizer finds the EASIEST path within your constraints
- If starting state is already valid, optimizer may do nothing
- To FORCE motion: make constraints that EXCLUDE the starting state (after t=0)

PRINCIPLE 7: Start conservative, fail fast
- Loose constraints -> optimization succeeds -> check if motion happened
- Tight constraints -> optimization fails -> you learn nothing
- Better to succeed with weak motion than fail completely

PRINCIPLE 8: Understand what you're actually constraining
- A lower bound applies at EVERY timestep, including the start
- Robot starts at {initial_height:.4f}m - if your lower bound exceeds this at t=0, optimization fails
- Think about the ENTIRE trajectory, not just the goal state

== AVAILABLE FUNCTIONS ==
vertcat, horzcat, mtimes, sin, cos, tan, sqrt, exp, log, fabs, fmax, fmin,
sum1, norm_2, atan2, asin, acos, tanh, MX, DM, cs.inf, pi, np

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


def create_repair_prompt(
    command: str,
    failed_code: str,
    error_message: str,
    attempt_number: int,
    initial_height: float = 0.2117,
) -> str:
    """
    Create a prompt to ask the LLM to fix failed MPC configuration code.

    Args:
        command: Original natural language command
        failed_code: The code that failed
        error_message: Error message from SafeExecutor/MPC
        attempt_number: Which attempt this is (1-10)
        initial_height: Robot initial COM height in meters

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
- CRITICAL: At k=0, bounds must INCLUDE the starting state (height={initial_height:.4f}m). Constraints that violate t=0 cause immediate failure!

Return ONLY corrected Python code."""
