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

You will receive detailed feedback after each iteration. Here's what each section means and how to use it:

--- MPC CONFIGURATION ---
Shows your current setup:
- Task name, duration, time step, horizon (number of timesteps)
- Number of constraints registered
- Contact phases with timing and patterns ([1,1,1,1]=grounded, [0,0,0,0]=flight)
USE IT: Verify your contact sequence matches the intended motion phases.

--- OPTIMIZATION STATUS ---
- SUCCESS: Solver found a feasible trajectory within your constraints
- FAILED: No feasible trajectory exists within your constraints
  - "Infeasibility": Constraints are mutually exclusive or violate physics
  - "Max iterations": Problem is too hard; simplify constraints
  - Check if constraints at k=0 allow the initial state (height={initial_height:.4f}m)
USE IT: If failed, LOOSEN constraints or check for constraint conflicts. Start simpler.

--- SIMULATION STATUS ---
- SUCCESS with tracking error: How well the robot followed the planned trajectory
  - tracking_error < 0.1: Excellent tracking
  - tracking_error 0.1-0.3: Acceptable
  - tracking_error > 0.3: Planned trajectory may be unrealistic
- FAILED: Robot fell, diverged, or hit physical limits
USE IT: High tracking error means the planned trajectory is hard to execute physically.

--- TASK PROGRESS TABLE ---
Shows % completion for each goal criterion (e.g., height, rotation):
- 0-25%: Constraint is too weak or wrong direction
- 25-75%: Constraint is working but needs to be stronger
- 75-99%: Almost there, minor tuning needed
- 100%: This criterion is satisfied, focus on others
USE IT: Focus on the LOWEST percentage criterion first. Strengthen that constraint.

--- TRAJECTORY METRICS ---

POSITION:
- initial/max/min/final height: Track vertical motion through the trajectory
- height_gain = max_height - initial_height (how high the robot jumped)
  - Good jump: 0.2-0.4m gain
  - Weak jump: <0.1m gain (strengthen upward velocity or height constraints)
- X/Y displacement: Horizontal drift (should be ~0 for pure vertical motions)
- total_distance: Path length traveled by COM
USE IT: If height_gain is low, robot isn't jumping high enough. Add stronger vertical velocity constraints during takeoff.

VELOCITY:
- max_com_velocity: Peak speed achieved (m/s)
  - Normal motion: 1-3 m/s
  - Aggressive motion: 3-5 m/s
- final_com_velocity: Speed at end (should be ~0 for stable landing)
- max_angular_velocity: Peak rotation speed (rad/s)
  - Backflip needs ~6-10 rad/s pitch rate
  - 180 degree turn needs ~3-6 rad/s yaw rate
- max_acceleration: Peak acceleration (m/s squared)
USE IT: If rotation is too slow, robot doesn't have enough angular momentum. Increase angular velocity constraints.

ORIENTATION (in radians AND degrees):
- roll: Rotation around X-axis (side-to-side tilt)
  - max_roll: Peak roll angle reached
  - total_roll_change: Final - initial roll
- pitch: Rotation around Y-axis (forward/backward tilt)
  - Backflip needs total_pitch_change of about 2*pi (360 degrees) or -2*pi
  - Front flip: positive pitch change
  - Back flip: negative pitch change
- yaw: Rotation around Z-axis (turning left/right)
  - 180 degree turn needs max_yaw of about pi (3.14 rad)
  - 90 degree turn needs max_yaw of about pi/2 (1.57 rad)
USE IT: Compare achieved rotation to target. If total_pitch_change is 1.5 rad but you need 6.28 rad (360 degrees), your pitch constraint is ~4x too weak.

TIMING:
- duration: Total trajectory time
- flight_duration: Time with all feet off ground
  - Good jump: 0.3-0.6s flight
  - Flip needs: 0.4-0.8s flight (enough time to rotate)
- flight_start_time: When takeoff occurred
USE IT: If flight_duration is too short, robot can't complete rotations. Extend flight phase in contact sequence.

JOINTS:
- max_joint_range: Largest joint motion (rad)
- avg_joint_range: Average joint motion
USE IT: If joints barely move, the motion is too conservative.

--- PHASE ANALYSIS ---

PRE-FLIGHT STANCE:
- duration: Time spent preparing to jump
- crouch_depth: How much the robot compressed (cm) - deeper = more power
  - Good crouch: 3-6cm
- takeoff_velocity: Vertical velocity at liftoff (m/s)
  - Good takeoff: 1.5-2.5 m/s upward
  - Weak takeoff: <1.0 m/s (won't get enough height)
- peak_GRF: Ground reaction force during push-off (Newtons)
  - Strong takeoff: 2-3x body weight
USE IT: If takeoff_velocity is low, robot isn't pushing hard enough. Strengthen vertical velocity constraints.

FLIGHT:
- duration: Time airborne
- peak_height: Maximum COM height reached
- roll/pitch/yaw_rate: Average angular velocity during flight (rad/s)
  - Backflip needs avg_pitch_rate of about 8-12 rad/s
- roll/pitch/yaw_change: Total rotation during flight
USE IT: This is where rotations happen. If pitch_change is low but pitch_rate is high, flight is too short.

LANDING:
- duration: Time in landing phase
- impact_velocity: Vertical speed at touchdown (m/s)
  - Safe landing: <1.5 m/s
  - Hard landing: >2.0 m/s (might fail simulation)
- impact_GRF: Ground reaction force at landing
USE IT: High impact_velocity means robot is coming down too fast. May need to constrain final velocity.

--- GROUND REACTION FORCES ---
- max_total_GRF: Peak force (Newtons)
- max_GRF_ratio: Peak force as multiple of body weight
  - Normal standing: ~1x body weight
  - Jump takeoff: 2-3x body weight
  - Hard landing: 3-5x body weight
- GRF_at_takeoff: Force right before liftoff
- GRF_at_landing: Force at touchdown
- left_right_asymmetry: Balance between legs (%)
  - <10%: Symmetric, good
  - >20%: Unbalanced, may cause rotation issues
USE IT: If takeoff GRF is only 1x body weight, robot isn't pushing hard enough to jump.

--- ACTUATOR STATUS ---
- torque_utilization: % of motor torque limit used
  - <70%: Robot has headroom, can push harder
  - 70-90%: Good utilization
  - >90%: At physical limits, can't do more
- torque_clipping: % of time motors were saturated
  - >5%: Motors are limiting performance
- velocity_utilization: % of joint speed limit used
- saturated_joints: Which joints hit limits most often
USE IT: If utilization is <50%, robot could move more aggressively. If >90%, you're asking for more than physically possible.

--- VS PREVIOUS ITERATION ---
Shows changes from last iteration with arrows (up=improved, down=regressed, same=no change):
- height_gain: Did the robot jump higher or lower?
- pitch/yaw/roll rotation: Did rotation increase or decrease?
USE IT: If a metric regressed, your last change made things worse. Revert or try different approach.

--- VISUAL FRAMES ---
Side-by-side comparison of PLANNED trajectory (from optimizer) vs SIMULATED trajectory (physics simulation).
USE IT: If they differ significantly, the planned trajectory is unrealistic.

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
