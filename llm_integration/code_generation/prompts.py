"""Prompt templates for LLM constraint generation."""

from __future__ import annotations

from typing import Any

import go2_config


def get_robot_details() -> dict[str, Any]:
    """Return robot-specific details from go2_config.

    Returns:
        Dictionary with mass, initial_height, joint_limits, geometry,
        and capability fields — all from go2_config.
    """
    return {
        "mass": go2_config.composite_mass,
        "initial_height": float(go2_config.initial_crouch_qpos[2]),
        "joint_limits_lower": go2_config.urdf_joint_limits_lower.tolist(),
        "joint_limits_upper": go2_config.urdf_joint_limits_upper.tolist(),
        "grf_limits": float(go2_config.grf_limits),
        "joint_velocity_limits": go2_config.urdf_joint_velocities.tolist(),
        "mu_ground": go2_config.experiment.mu_ground,
        "body_half_extents": go2_config.body_half_extents,
        "leg_lengths": go2_config.leg_lengths,
        "hip_spacing": go2_config.hip_spacing,
        "capability_limits": go2_config.capability_limits,
    }


def get_system_prompt() -> str:
    """Get the system prompt that instructs the LLM on MPC configuration + constraint generation.

    Returns:
        System prompt string with accurate physical parameters from go2_config.
    """
    details = get_robot_details()
    mass = details["mass"]
    initial_height = details["initial_height"]
    grf_limit = float(details["grf_limits"])
    jvel_limit = float(max(details["joint_velocity_limits"]))
    mu = float(details["mu_ground"])
    bhe = details["body_half_extents"]
    ll = details["leg_lengths"]
    hs = details["hip_spacing"]
    cl = details["capability_limits"]
    mpc_dt = go2_config.mpc_config.mpc_dt

    base = f"""You are a robotics expert generating MPC configurations for quadruped robot trajectory optimization.

== 1. ROBOT (Unitree Go2) ==

Mass: {mass:.2f} kg | Initial COM height: {initial_height:.4f}m
Body trunk: ~{bhe[0] * 2 * 100:.1f}cm long x {bhe[1] * 2 * 100:.1f}cm wide x {bhe[2] * 2 * 100:.1f}cm tall (model half-extents: {bhe[0]}/{bhe[1]}/{bhe[2]}m)
Leg length: ~{ll["total"] * 100:.0f}cm fully extended ({ll["thigh"] * 100:.1f}cm thigh + {ll["calf"] * 100:.1f}cm calf)
Hip spacing: Front/rear ~{hs["front_rear_from_com"] * 100:.0f}cm from COM, Left/right ~{hs["left_right_from_com"] * 100:.0f}cm from COM
Joint limits (from URDF, per joint in rad): lower={details["joint_limits_lower"][:3]}, upper={details["joint_limits_upper"][:3]} (FL; pattern repeats per leg with RL/RR thigh having different range)
Per-component GRF limit: {grf_limit:.0f} N (fx,fy,fz each per foot) | Joint velocity limit: {jvel_limit:.1f} rad/s
Ground friction coefficient: {mu}

Physical capability limits (do NOT exceed):
- Max COM height gain: ~{cl["min_height_gain_normal"]}-{cl["max_height_gain_normal"]}m (normal jump), ~{cl["max_height_gain_aggressive"]}m (aggressive)
- Max takeoff vz: ~{cl["min_takeoff_vz"]}-{cl["max_takeoff_vz"]} m/s | Max flight duration: ~{cl["min_flight_duration"]}-{cl["max_flight_duration"]}s
- Max peak GRF: ~{cl["min_peak_grf_bodyweight_multiple"]:.0f}-{cl["max_peak_grf_bodyweight_multiple"]:.0f}x body weight (~{cl["min_peak_grf_total"]:.0f}-{cl["max_peak_grf_total"]:.0f} N total across 4 feet)
- Max COM acceleration: ~{cl["min_com_accel_typical_g"]:.0f}-{cl["max_com_accel_typical_g"]:.0f}g typical, up to ~{cl["max_com_accel_feasible_g"]:.0f}g solver-feasible
- Peak angular velocity: {cl["min_peak_angular_velocity"]:.0f}-{cl["peak_angular_velocity"]:.0f} rad/s
- vz_takeoff = g * flight_duration / 2, but flight_duration MUST be <= {cl["max_flight_duration"]}s

Key physics:
- The MPC uses FULL-BODY DYNAMICS (not simplified centroidal). Forces couple through
  Jacobians — leg configuration affects how GRFs translate to COM acceleration.
  The mass matrix is configuration-dependent (changes as joints move).
- Torque feasibility is enforced at 80% of motor limits to leave headroom for the
  PD tracking controller. Design GRFs conservatively — don't max out forces.
- The solver is computationally heavy — keep constraints conservative and prefer
  wider bounds. Smooth reference trajectories help convergence.
- Rotation: angle_change = angular_velocity × time
- Projectile motion: peak_height = initial_height + v²/(2g)
- Angular momentum is STRICTLY CONSERVED during flight — the solver enforces this
  as a hard constraint (angular_momentum_flight_constraint). ALL rotational momentum
  must be generated during stance (before takeoff). During flight, joint motion can
  redistribute angular momentum between base and legs but CANNOT create new angular
  momentum. Plan takeoff GRFs and timing to generate sufficient angular velocity
  BEFORE the feet leave the ground.
- All forces during stance go through the feet and must satisfy the friction box
  (|fx| <= mu*fz AND |fy| <= mu*fz per foot). The per-phase breakdown in trajectory metrics
  shows what % of each motion occurs during stance vs flight — cross-reference this
  with friction cone violations to identify motion the solver is placing in the
  wrong phase. Wide constraint bounds during ground phases let the optimizer cheat.

== 2. STATE VECTOR & INDEX CONSTANTS ==

Both constraint functions and reference trajectories use a 30-dim state vector:
  x_k[0:3]   COM position [x, y, z] (meters)
  x_k[3:6]   COM velocity [vx, vy, vz] (m/s)
  x_k[6:9]   orientation [roll, pitch, yaw] (radians)
  x_k[9:12]  angular velocity [wx, wy, wz] (rad/s)
  x_k[12:24] joint angles (12 joints)
  x_k[24:30] integral states (used internally by MPC cost — ignore in constraints)

Pre-imported index constants:
  STATES_DIM = 30, INPUTS_DIM = 24
  IDX_POS       = slice(0, 3)     # COM position
  IDX_VEL       = slice(3, 6)     # COM velocity
  IDX_EULER     = slice(6, 9)     # Euler angles
  IDX_ANG_VEL   = slice(9, 12)    # Angular velocity
  IDX_JOINTS    = slice(12, 24)   # Joint angles
  IDX_INTEGRALS = slice(24, 30)   # Integral states (set to 0)
  IDX_PITCH     = 7               # Scalar pitch index

Input vector (u_k) index constants:
  IDX_U_JOINT_VEL = slice(0, 12)  # u_k: joint velocities
  IDX_U_GRF       = slice(12, 24) # u_k: GRF (4 legs × 3)
  IDX_GRF_Z       = [14, 17, 20, 23]  # u_k: GRF z-component per foot

== 3. MPC API ==

You are tasked to write a script that configures the MPC for the given task:
  Assume that you are given a global variable called `mpc`.
  Remember that this is a script, so if you define a function as an entrypoint, you need to invoke it yourself at the end of the script.

Include these calls (items 4-6 are REQUIRED — solver FAILS without them):
  1. mpc.set_task_name("...")              # descriptive name (defaults to "unknown")
  2. mpc.set_duration(seconds)            # (defaults to {go2_config.mpc_config.duration})
     Prefer the shortest duration that fits the motion — fewer timesteps make
     the solver's job easier. Start short, extend only after convergence.
  3. mpc.set_time_step({mpc_dt})              # defaults to {mpc_dt}
  4. mpc.set_contact_sequence(array)       # ← REQUIRED — solver FAILS without this
  5. mpc.add_constraint(constraint_func)   # ← REQUIRED — solver FAILS without this
  6. mpc.set_reference_trajectory(func)    # ← REQUIRED — solver FAILS without this

Contact patterns (EVERY motion needs one, including ground-based like squatting):
  [1,1,1,1] = all feet grounded | [0,0,0,0] = flight phase
  Build via: mpc._create_phase_sequence([(name, duration_sec, [FL,FR,RL,RR]), ...])
  IMPORTANT: Call set_duration() and set_time_step() BEFORE _create_phase_sequence(),
  because it uses the current duration and dt to compute the contact array size.
  Phase durations should sum to the total duration set via mpc.set_duration().
  If they sum to less, remaining timesteps default to all-feet-grounded. If more,
  excess phases are silently truncated.

Optional: mpc.set_slack_weights({{"your_constraint_func_name": weight, ...}})
  Controls how strictly YOUR constraints are enforced (higher = harder).
  Default weight for your constraints: 1e3. Never go below 1e3 — smaller values
  mean the solver barely enforces your constraints. Physics constraints are always hard
  — you cannot soften them. Hard constraint names: friction_cone_constraints,
  foot_height_constraints, no_slip_constraints, joint_limits_constraints,
  input_limits_constraints, body_clearance_constraints, link_clearance_constraints,
  torque_feasibility_constraints, angular_momentum_flight_constraint,
  joint_acceleration_constraint.
  IMPORTANT: Do NOT name your constraint functions with any of these names, or they
  will be treated as hard constraints (no slack) and solver failures become likely.

== 4. CONSTRAINT DESIGN ==

Signature:
  def name(x_k, u_k, kindyn_model, config, contact_k, k, horizon):
      progress = k / horizon  # ranges from 1/horizon to (horizon-1)/horizon (never exactly 0 or 1)
      return (constraint_expr, lower_bound, upper_bound)  # CasADi MX

Parameters:
  x_k: CasADi MX (30,) — state at timestep k (see STATE VECTOR above, use indices 0-23)
  u_k: CasADi MX (24,) — control input at timestep k
    u_k[0:12]  = joint velocities [rad/s] (3 per leg: hip, thigh, calf × FL, FR, RL, RR)
    u_k[12:24] = ground reaction forces [N] (3 per leg: fx, fy, fz × FL, FR, RL, RR)
  kindyn_model: robot kinematics/dynamics model with forward kinematics and Jacobian
    functions for each foot (e.g. kindyn_model.forward_kinematics_FL_fun(H, joints))
    — rarely needed, only for foot-position-based constraints
  config: robot configuration object — access physical params via config.robot_data.mass,
    config.robot_data.grf_limits, config.experiment.mu_ground, etc.
    — rarely needed, physical limits are already in this prompt
  contact_k: CasADi MX (4,) — symbolic foot contact flags at timestep k
    (do NOT use for if/else bounds — see P1)

Constraint application range:
  YOUR constraints are applied at k=1 through k=horizon-1 ONLY (never at k=0 or
  k=horizon). State-only system constraints (joint_limits, foot_height, body_clearance, link_clearance)
  are additionally applied at k=horizon by the solver using a hardcoded name check.
  YOUR constraints are NEVER applied at k=horizon regardless of whether they use u_k.
  The initial state (k=0) is enforced separately by the solver. Design
  terminal constraints to tighten as progress approaches 1.0 — the last applied
  step is k=horizon-1 (progress = (horizon-1)/horizon, e.g. 0.98 for horizon=50).

P1 — SMOOTH BOUNDS: Bounds must be continuous across timesteps. NEVER use if/else
  branches, contact_k-based step functions, or anything that creates sudden jumps.
  Use progress-based ramps: bound = start + progress * change. For phase-specific
  behavior, use Gaussians: exp(-((progress - center)/width)²).
  IMPORTANT: Python if/else on CasADi symbolic variables (e.g. if x_k[2] > 0.3:)
  silently converts the symbolic expression to a concrete boolean — this does NOT
  create a conditional constraint. Use cs.if_else(condition, true_val, false_val)
  for symbolic branching.

P2 — COMPATIBLE WITH INITIAL STATE: Robot starts at height={initial_height:.4f}m,
  angles=0, velocities=0. Your constraints start at k=1, but bounds at k=1 must be
  reachable from the initial state in one timestep. Bounds that jump far from the
  initial state at early timesteps cause solver failures.

P3 — DON'T OVER-CONSTRAIN: Use one-sided bounds with large finite numbers (lower, 1e6)
  or (-1e6, upper) when possible. NEVER use cs.inf or np.inf in bounds — use 1e6 / -1e6
  instead, as infinite values cause numerical failures in the solver. Constrain RATES not positions for dynamic motions. Constraints define
  FEASIBLE REGIONS, not exact trajectories — the optimizer finds the optimal path.
  To FORCE motion, exclude the starting state (after t=0). DON'T add more constraints
  to fix failures — more constraints = smaller feasible region = harder problem.

P4 — TERMINAL CONSTRAINTS: The base MPC does NOT enforce terminal state requirements.
  For safe landing, tighten bounds as progress→1.0 (small velocities, stable orientation).
  Use fmax(0, (progress - 0.8) / 0.2) for smooth late-activation ramps.

P5 — ONE CONSTRAINT PER VARIABLE: Don't constrain the same variable in multiple
  functions — ALL bounds apply at ALL timesteps, so the tightest bound wins everywhere.
  Combine phase-specific logic into ONE function per variable. Don't tighten constraints
  after failures (violations mean the feasible region is already too small).

Constraint hardness guidance (for iteration 2+):
  - YOUR constraints with large slack: bounds too tight, widen at violated timesteps
  - SYSTEM constraints with large slack: motion is physically difficult, adjust timing
  - body_clearance/link_clearance violations mean too much rotation during ground contact
  - Use mpc.set_slack_weights() to prioritize which constraints matter most

Solver failure causes (Invalid_Number = NaN, infeasible problem):
  - Bounds that demand states unreachable given the current contact mode
  - Conflicting constraints that leave no feasible region
  - Fix by widening bounds 2-3×, NOT by removing constraints entirely — removing
    all guidance produces degenerate solutions

Body-link ground clearance:
  - The solver constrains foot heights, COM-based body clearance, AND individual
    link heights (calves and head) to stay above ground. If link_clearance
    violations appear, the motion requires too much rotation during ground contact.

== 5. REFERENCE TRAJECTORY ==

Signature:
  def generate_reference_trajectory(initial_state, horizon, contact_sequence, mpc_dt, robot_mass):
      # initial_state: np.ndarray (30,) | horizon: int | contact_sequence: np.ndarray (4, horizon)
      # mpc_dt: float (seconds) | robot_mass: float (kg)
      # Returns: (X_ref, U_ref) — shapes (30, horizon+1) and (24, horizon)
  mpc.set_reference_trajectory(generate_reference_trajectory)

How to build:
  1. Extract initial state: pos0 = initial_state[IDX_POS].copy(), eul0 = initial_state[IDX_EULER].copy(), etc.
  2. Count timesteps per phase from contact_sequence columns
  3. Loop over each phase, filling X_ref[:, k] and U_ref[:, k] per timestep

Physics rules:
  - Velocities must be consistent with positions
  - Flight phases: ballistic z(t) = z0 + vz0*t - 0.5*g*t², GRF=0, angular momentum
    strictly conserved (enforced by solver)
  - Stance phases: GRF_z per grounded foot ≈ robot_mass * g / n_grounded_feet
  - Use smooth interpolation (e.g. 10t³ - 15t⁴ + 6t⁵) for transitions
  - Integrate angles from angular velocities — don't set angles without matching omega
  - Set IDX_INTEGRALS to 0.0, IDX_JOINTS to initial joint angles unless needed
  - GRF z-component for foot i at IDX_GRF_Z[i]

Constraint-reference interplay:
  - Constraints define the FEASIBLE REGION; reference is the COST TARGET and INITIAL GUESS
  - The solver's cost function penalizes deviation from the reference at EVERY timestep
  - If constraints force rotation, the reference MUST show that rotation or the cost fights it
  - Reference must be INSIDE constraint bounds (ideally near the center)
  - Design them TOGETHER — when you change constraints, update the reference to match

== 6. AVAILABLE FUNCTIONS ==

CasADi (usable directly or via cs.*):
  Math: sin, cos, tan, asin, acos, atan2, sqrt, exp, log, fabs, tanh, sinh, cosh
  Comparison: fmax, fmin, if_else, logic_and, logic_or
  Linear algebra: vertcat, horzcat, mtimes, dot, cross, norm_2, sum1, transpose, inv
  Matrix ops: reshape, repmat, eye, zeros, ones
  Types: MX, SX, DM, cs.inf

NumPy (via np.*):
  np.array, np.zeros, np.ones, np.eye, np.concatenate, np.stack, np.linalg,
  np.maximum, np.minimum, np.pi, np.sin, np.cos, np.sqrt, np.inf

Python builtins: abs, sum, round, max, min, len, range, enumerate, zip, float,
  int, str, list, tuple, dict, bool, hasattr, getattr, isinstance, type, print
Constants: pi, inf
Rotation: SO3 (from liecasadi)

== 7. ITERATION STRATEGY ==

ITERATION 1: Minimal viable constraints
  - Maximum 2-3 constraints, bounds 2-3x wider than you think necessary
  - Goal: Solver converges, motion happens (even if imperfect)

LATER ITERATIONS: You receive scores for ALL past iterations, plus 3 sampled
  iterations (the best + 2 random, weighted by score) with their constraint code
  and detailed performance summaries. Learn from ALL of them — high scores show
  what works, low scores show what to avoid and why. You decide whether to tweak
  a good approach or pivot to something new.

  If the solver is failing → first try shorter duration and wider bounds
  (simplify the problem). A converged solution with imperfect task completion
  is far more valuable than an unconverged one.

== 8. TASK ==
Generate MPC configuration and constraints for the requested behavior.
Think about: What motion is needed? What constraints will FORCE that motion?"""

    return base


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
2. What contact sequence is appropriate?
3. What physical quantities need to be constrained to achieve this?
4. What are reasonable bounds that FORCE the desired motion?

Return ONLY Python code."""


def create_repair_prompt(
    command: str,
    failed_code: str,
    error_message: str,
    attempt_number: int,
    mpc_dt: float | None = None,
) -> str:
    """Create a prompt to ask the LLM to fix failed MPC configuration code.

    Args:
        command: Original natural language command.
        failed_code: The code that failed.
        error_message: Error message from SafeExecutor/MPC.
        attempt_number: Which attempt this is (1-10).
        mpc_dt: The LLM's current time step. Falls back to base config only
            on iteration 1 (before any LLM has changed dt).

    Returns:
        Repair prompt string.
    """
    details = get_robot_details()
    initial_height = details["initial_height"]
    if mpc_dt is None:
        mpc_dt = go2_config.mpc_config.mpc_dt
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

⚠️ MANDATORY CHECKLIST - Items 4-6 are REQUIRED (solver FAILS without them):
□ mpc.set_task_name("...")              (recommended, defaults to "unknown")
□ mpc.set_duration(...)                 (recommended, defaults to {go2_config.mpc_config.duration})
□ mpc.set_time_step({mpc_dt})              (recommended, defaults to {mpc_dt})
□ mpc.set_contact_sequence(...)  ← REQUIRED - THIS IS THE #1 MISSING CALL
□ mpc.add_constraint(...)        ← REQUIRED
□ mpc.set_reference_trajectory(...)  ← REQUIRED - cost target AND initial guess

Other requirements:
- Constraint function must have 7 parameters: (x_k, u_k, kindyn_model, config, contact_k, k, horizon)
- Must return exactly 3 values: (constraint_expr, lower_bound, upper_bound)
- All return values must be CasADi MX expressions (use vertcat for multiple constraints)
- YOUR constraints are applied at k=1 through k=horizon-1 (NOT at k=0 or k=horizon). State-only system constraints (joint_limits, foot_height, body_clearance, link_clearance) are also applied at k=horizon. Bounds at k=1 must be reachable from the starting state (height={initial_height:.4f}m).

Return ONLY corrected Python code."""
