"""Prompt templates for LLM constraint generation."""

from __future__ import annotations

from typing import Any


def get_robot_details(config: Any = None) -> dict[str, Any]:
    """Extract robot-specific details from config or use sensible defaults.

    Args:
        config: Optional robot configuration object.

    Returns:
        Dictionary with mass, initial_height, joint_limits_lower, joint_limits_upper.
    """
    details: dict[str, Any] = {
        "mass": 15.0,
        "initial_height": 0.2117,
        "joint_limits_lower": [-0.8, -1.6, -2.6] * 4,
        "joint_limits_upper": [0.8, 1.6, -0.5] * 4,
    }

    if config is None:
        return details

    details["mass"] = float(config.robot_data.mass)
    details["initial_height"] = float(config.experiment.initial_qpos[2])
    details["joint_limits_lower"] = config.robot_data.joint_limits_lower.tolist()
    details["joint_limits_upper"] = config.robot_data.joint_limits_upper.tolist()

    return details


def get_system_prompt(config: Any = None) -> str:
    """Get the system prompt that instructs the LLM on MPC configuration + constraint generation.

    Args:
        config: Optional robot configuration object.

    Returns:
        System prompt string with accurate physical parameters.
    """
    details = get_robot_details(config)
    mass = details["mass"]
    initial_height = details["initial_height"]

    base = f"""You are a robotics expert generating MPC configurations for quadruped robot trajectory optimization.

OUTPUT FORMAT: Return ONLY Python code. You may use ```python code blocks.

== CRITICAL: MANDATORY REQUIREMENTS ==

YOUR CODE WILL FAIL WITHOUT ALL SIX OF THESE CALLS:

1. mpc.set_task_name("...")           <- REQUIRED
2. mpc.set_duration(...)              <- REQUIRED
3. mpc.set_time_step(0.02)            <- REQUIRED
4. mpc.set_contact_sequence(...)      <- REQUIRED - #1 CAUSE OF FAILURES
5. mpc.add_constraint(...)            <- REQUIRED
6. mpc.set_reference_trajectory(...)  <- REQUIRED - provides solver warmstart

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

Optional calls:
  mpc.set_slack_weights({{"constraint_name": weight, ...}})
    # Adjust how strictly each constraint type is enforced
    # Higher weight = harder constraint (solver avoids violating it)
    # Lower weight = softer constraint (solver may relax it for feasibility)
    # Default weights: friction_cone=1e5, foot_height=1e4, complementarity=1e2
    # Example: mpc.set_slack_weights({{"contact_aware_constraint": 1e4}})

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

PRINCIPLE 5: Constrain RATES for dynamic motions, not position trajectories
- Velocity/rate constraints let the optimizer find natural motion paths
- Position trajectory constraints often conflict with dynamics
- Define WHERE you want to end up, not HOW to get there

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

PRINCIPLE 9: YOU must specify terminal constraints
- The base MPC does NOT enforce any terminal state requirements
- For safe landing, you need bounds that TIGHTEN as progress approaches 1.0:
  - Terminal velocities (vx, vy, vz) should be small for stable landing
  - Terminal angular velocities (wx, wy, wz) should be small
  - Terminal orientation depends on the task
- Use fmax to create smooth late-activation: fmax(0, (progress - threshold) / width) gives a ramp from 0 to 1
  - Example: activation = fmax(0, (progress - 0.8) / 0.2) is 0 until progress=0.8, then ramps to 1
- Without terminal constraints, the robot may land in unstable configurations

== CRITICAL: AVOID CONTACT-BASED STEP FUNCTIONS ==

DO NOT use contact_k to create step-function bounds like:
  upper = (1.0 - contact_k[0]) * X + contact_k[0] * Y  # BAD

This creates SHARP DISCONTINUITIES at phase boundaries that the optimizer struggles with.

INSTEAD, always use progress-based bounds with SMOOTH functions (polynomials, quadratics, or Gaussians).

When using Gaussian dips (exp(-((progress - center)/width)^2)) to force behavior during a phase:
- Make the constraint STRONG ENOUGH to actually force the motion (weak constraints get ignored)
- But keep bounds PHYSICALLY ACHIEVABLE - check the physics limits in this prompt and don't exceed them
- For velocity constraints, "excluding zero" means one bound must cross zero:
  - To force POSITIVE velocity: make LOWER bound positive (> 0)
  - To force NEGATIVE velocity: make UPPER bound negative (< 0)
- Consider adding supporting constraints (height, terminal velocity) to help guide the optimizer

== CONSTRAINT ANTI-PATTERNS (COMMON FAILURES) ==

1. DON'T constrain the same variable in multiple constraint functions
   - Even if constraints are meant for different phases, ALL bounds apply at ALL timesteps
   - If one constraint has upper=5.0 and another has upper=2.0 on the same variable, the tighter bound (2.0) wins everywhere
   - Combine phase-specific logic into ONE constraint function per variable

2. DON'T use if/else that creates discontinuous bounds
   - Solver needs smooth feasible regions
   - Use linear ramps: bound = start + progress * change

3. DON'T tighten constraints after failures
   - Constraint violations mean the feasible region is TOO SMALL
   - Each failed iteration should LOOSEN bounds, not tighten

4. DON'T add more constraints to fix failures
   - More constraints = smaller feasible region = harder problem
   - If failing with N constraints, try N-2 constraints

5. DON'T script the trajectory
   - You define WHERE the robot should end up, not HOW it gets there
   - The optimizer finds the optimal path within your bounds

== PHYSICS FACTS ==

Angular motion:
- Full rotation = 2π radians ≈ 6.28 rad
- rotation_angle = angular_velocity × time
- Angular momentum is conserved during flight (no external torques)
- All rotation must happen during flight phase ([0,0,0,0] contact)

Achievable ranges for this robot:
- Peak angular velocity: 8-15 rad/s (physically realistic)
- Flight duration: 0.4-0.8s (based on achievable jump height)

== ITERATION STRATEGY ==

ITERATION 1: Minimal viable constraints
  - Maximum 2-3 constraints
  - Bounds should be 2-3x wider than you think necessary
  - Goal: Solver converges, motion happens (even if imperfect)

AFTER A FAILURE:
  - REMOVE constraints (fewer = easier)
  - WIDEN bounds (larger feasible region)
  - Try constraining DIFFERENT quantities

AFTER A SUCCESS WITH WEAK MOTION:
  - Tighten only ONE bound by 10-20%
  - Never tighten multiple constraints simultaneously

The feedback loop will guide refinement. Your job is to keep the
problem SOLVABLE while steering toward the goal.

== AVAILABLE FUNCTIONS ==
vertcat, horzcat, mtimes, sin, cos, tan, sqrt, exp, log, fabs, fmax, fmin,
sum1, norm_2, atan2, asin, acos, tanh, MX, DM, cs.inf, pi, np

== CONSTRAINT-REFERENCE INTERPLAY ==

Constraints and reference trajectories work TOGETHER but serve different roles:

1. CONSTRAINTS define the FEASIBLE REGION — the set of trajectories the solver is allowed to explore.
   If constraints have loopholes, the solver will exploit them regardless of the reference.

2. REFERENCE TRAJECTORY provides the INITIAL GUESS — where the solver starts searching.
   A good reference helps the solver converge faster and find better local minima.

Key interactions:
- If constraints force rotation, the reference MUST show that rotation (otherwise solver
  starts far from the feasible region and struggles to converge)
- If the reference is outside constraint bounds, the solver may fail immediately
- The reference should sit roughly in the CENTER of the constraint bounds
- Design them TOGETHER — when you change constraints, update the reference to match

== UNDERSTANDING FEEDBACK ==

Feedback is provided starting from iteration 2 onward. On iteration 1 (first attempt),
you only receive the task command with no feedback sections.

You will receive structured feedback with these sections:

--- TERMINOLOGY ---
Definitions of SOLVER CONVERGED, SOLVER FAILED, and Score labels used throughout.

--- ITERATION HISTORY ---
Summaries of past iterations (NOT including the current one) with approach, feedback,
simulation, and key metrics. Use this to avoid repeating failed strategies.

--- MODE USED FOR THIS ITERATION: TWEAK / PIVOT / INITIAL ---
What mode was used to generate the current iteration's code:
- TWEAK: Incremental improvements were requested
- PIVOT: A fundamentally different strategy was requested (approach was stagnating)
- INITIAL: First iteration

--- CURRENT ITERATION DETAILED ANALYSIS [SOLVER STATUS] Score: X.XX ---
The current iteration's solver status and score are shown in this header.

--- METRICS FOR THIS ITERATION ---
Full trajectory metrics, constraint hardness analysis, reference RMSE, and solver status.

--- ENTIRE CODE FOR THIS ITERATION ---
Your full code (both constraint and reference trajectory functions together).

--- ENTIRE FEEDBACK FOR THIS ITERATION ---
Unified LLM analysis covering both constraint design and reference trajectory:
- Which constraints are working/failing and specific bound changes needed
- Reference trajectory physics plausibility, RMSE, and phase timing
- How constraints and reference interact
- Prioritized action items

== CONSTRAINT HARDNESS ANALYSIS ==

After each solve, you will receive a CONSTRAINT HARDNESS ANALYSIS section in the feedback.
This uses a slack formulation to measure how difficult each constraint is to satisfy.

You will receive raw slack values and violation timesteps. YOU must assess the severity of each
violation from the raw numbers — there are no pre-classified severity labels. Larger slack values
mean the solver had to relax the constraint more to find a solution.

How to use this information:
1. If YOUR constraints (contact_aware_constraint) show CRITICAL slack:
   - Your bounds are too tight for the requested motion
   - Widen bounds at the timesteps shown in the "Violated at" field
   - Use phase-aware bounds (different for stance vs flight)
2. If SYSTEM constraints show CRITICAL slack:
   - The motion itself is physically difficult
   - Adjust contact_sequence timing or motion parameters
   - body_clearance violations often mean too much rotation during ground contact
3. You can adjust constraint priority with mpc.set_slack_weights():
   - Lower weight = solver relaxes that constraint more easily
   - Higher weight = solver tries harder to satisfy it
   - Use this to prioritize which constraints matter most for your motion

== REFERENCE TRAJECTORY (MANDATORY) ==

You MUST always define a reference trajectory function. This provides a physics-informed
initial guess (warmstart) that helps the solver converge much faster and find better
solutions. The MPC cost function and slack variables remain unchanged — the reference
trajectory is used ONLY as the solver's starting point.

YOUR CODE WILL FAIL WITHOUT THIS CALL:
6. mpc.set_reference_trajectory(...)   <- REQUIRED

Function signature:
  def generate_reference_trajectory(initial_state, horizon, contact_sequence, mpc_dt, robot_mass):
      # initial_state: np.ndarray (30,) — current robot state
      # horizon: int — number of MPC timesteps
      # contact_sequence: np.ndarray (4, horizon) — foot contact pattern
      # mpc_dt: float — time step in seconds
      # robot_mass: float — mass in kg
      # Returns: (X_ref, U_ref)
      #   X_ref: np.ndarray shape (30, horizon+1)
      #   U_ref: np.ndarray shape (24, horizon)

Register it:
  mpc.set_reference_trajectory(generate_reference_trajectory)

PRE-IMPORTED INDEX CONSTANTS (use these instead of raw numbers):
  # Dimensions
  STATES_DIM = 30          # state vector length
  INPUTS_DIM = 24          # input vector length

  # State slices (for X_ref rows)
  IDX_POS       = slice(0, 3)     # COM position [x, y, z]
  IDX_VEL       = slice(3, 6)     # COM velocity [vx, vy, vz]
  IDX_EULER     = slice(6, 9)     # Euler angles [roll, pitch, yaw]
  IDX_ANG_VEL   = slice(9, 12)    # Angular velocity [wx, wy, wz]
  IDX_JOINTS    = slice(12, 24)   # Joint angles (12 joints)
  IDX_INTEGRALS = slice(24, 30)   # Integral states (set to 0)
  IDX_PITCH     = 7               # Scalar pitch index (row 7)

  # Input slices (for U_ref rows)
  IDX_U_JOINT_VEL = slice(0, 12)  # Joint velocities
  IDX_U_GRF       = slice(12, 24) # Ground reaction forces (4 legs × 3)
  IDX_GRF_Z       = [14, 17, 20, 23]  # GRF z-component per foot (FL, FR, RL, RR)

== HOW TO BUILD A REFERENCE TRAJECTORY ==

Your function MUST build the trajectory phase-by-phase matching your contact sequence.
Determine the number of timesteps in each phase from the contact_sequence, then loop
through each phase filling X_ref and U_ref.

STRUCTURE:
  1. Extract initial state:  pos0 = x0[IDX_POS].copy(), eul0 = x0[IDX_EULER].copy(), etc.
  2. Count timesteps per phase from the contact_sequence columns
  3. Loop over each phase, filling X_ref[:, k] and U_ref[:, k] per timestep

PHYSICS RULES (apply to ALL motions):
  - Velocities must be consistent with positions (if position changes, velocity must be nonzero)
  - Flight phases (all contacts = 0): ballistic height z(t) = z0 + vz0*t - 0.5*g*t²,
    angular momentum conserved (constant angular velocity), GRF = 0
  - Stance phases (any contact = 1): GRF_z per grounded foot ≈ robot_mass * g / n_grounded_feet
  - Use smooth interpolation (e.g. 10t^3 - 15t^4 + 6t^5) for transitions between states
  - Integrate angles from angular velocities — don't set angles without matching omega
  - Set IDX_INTEGRALS rows to 0.0, set IDX_JOINTS to initial joint angles unless needed
  - GRF z-component for foot i is at index IDX_GRF_Z[i] (i.e. 14, 17, 20, 23)

Key points:
  - The reference trajectory gives the solver a good starting point, NOT a tracking target.
  - The existing phase-aware cost function + slack constraints are still used for optimization.
  - You still MUST provide constraints via mpc.add_constraint() — constraints enforce
    feasibility, the reference trajectory just helps the solver converge.
  - A physically realistic reference trajectory dramatically improves convergence.

== TASK ==
Generate MPC configuration and constraints for the requested behavior.
Think about: What motion is needed? What constraints will FORCE that motion?
Start with simple, loose constraints. The feedback loop will help you refine."""

    robot_context = f"""

ROBOT PHYSICAL DETAILS:
- Mass: ~{mass:.1f} kg
- Body: ~30cm x 20cm x 10cm
- Leg reach: ~30cm leg extension
- Joint limits: Hip: +/-45deg, Thigh: +/-90deg, Calf: +/-150deg
- Realistic jump height: ~0.5-0.8m
- Typical stance COM height: ~{initial_height:.2f}m
- Foot spacing: Front/rear: ~30cm, Left/right: ~20cm

Use these physical limits to create realistic constraints."""

    return base + robot_context


def get_user_prompt(command: str) -> str:
    """Create the initial user prompt from a natural language command.

    Args:
        command: Natural language command (e.g., "do a backflip").

    Returns:
        Formatted user prompt.
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
    config: Any = None,
) -> str:
    """Create a prompt to ask the LLM to fix failed MPC configuration code.

    Args:
        command: Original natural language command.
        failed_code: The code that failed.
        error_message: Error message from SafeExecutor/MPC.
        attempt_number: Which attempt this is (1-10).
        config: Optional robot configuration object.

    Returns:
        Repair prompt string.
    """
    initial_height = get_robot_details(config)["initial_height"]
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

⚠️ MANDATORY CHECKLIST - Your code MUST include ALL SIX of these:
□ mpc.set_task_name("...")
□ mpc.set_duration(...)
□ mpc.set_time_step(0.02)
□ mpc.set_contact_sequence(...)  ← THIS IS THE #1 MISSING CALL
□ mpc.add_constraint(...)
□ mpc.set_reference_trajectory(...)  ← REQUIRED - provides solver initial guess

Other requirements:
- Constraint function must have 7 parameters: (x_k, u_k, kindyn_model, config, contact_k, k, horizon)
- Must return exactly 3 values: (constraint_expr, lower_bound, upper_bound)
- All return values must be CasADi MX expressions (use vertcat for multiple constraints)
- CRITICAL: At k=0, bounds must INCLUDE the starting state (height={initial_height:.4f}m). Constraints that violate t=0 cause immediate failure!

Return ONLY corrected Python code."""
