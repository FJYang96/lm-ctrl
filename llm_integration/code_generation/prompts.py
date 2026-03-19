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
        "mass": 15.019,
        "initial_height": 0.2117,
        "joint_limits_lower": [
            -0.8,
            -1.57,
            -2.6,
            -0.8,
            -1.57,
            -2.6,
            -0.8,
            -0.52,
            -2.6,
            -0.8,
            -0.52,
            -2.6,
        ],
        "joint_limits_upper": [0.8, 1.6, -0.84] * 4,
    }

    if config is None:
        return details

    details["mass"] = float(config.robot_data.mass)
    details["initial_height"] = float(config.experiment.initial_qpos[2])
    details["joint_limits_lower"] = config.robot_data.joint_limits_lower.tolist()
    details["joint_limits_upper"] = config.robot_data.joint_limits_upper.tolist()

    if hasattr(config.robot_data, "grf_limits"):
        val = config.robot_data.grf_limits
        details["grf_limits"] = val.tolist() if hasattr(val, "tolist") else float(val)
    if hasattr(config.robot_data, "joint_velocity_limits"):
        val = config.robot_data.joint_velocity_limits
        details["joint_velocity_limits"] = (
            val.tolist() if hasattr(val, "tolist") else float(val)
        )
    if hasattr(config.experiment, "mu_ground"):
        details["mu_ground"] = float(config.experiment.mu_ground)

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

    # Extract GRF limit, joint velocity limit, and friction coefficient
    grf_limit = 500.0  # default
    grf_limits_raw = details.get("grf_limits")
    if grf_limits_raw is not None:
        if isinstance(grf_limits_raw, (int, float)):
            grf_limit = float(grf_limits_raw)
        else:
            grf_limit = (
                float(grf_limits_raw[2])
                if len(grf_limits_raw) > 2
                else float(max(grf_limits_raw))
            )

    jvel_limit = 10.0  # default
    jvel_limits_raw = details.get("joint_velocity_limits")
    if jvel_limits_raw is not None:
        if isinstance(jvel_limits_raw, (int, float)):
            jvel_limit = float(abs(jvel_limits_raw))
        else:
            jvel_limit = (
                float(max(abs(v) for v in jvel_limits_raw)) if jvel_limits_raw else 10.0
            )

    mu = 0.5  # default
    mu_raw = details.get("mu_ground")
    if mu_raw is not None:
        mu = float(mu_raw)

    base = f"""You are a robotics expert generating MPC configurations for quadruped robot trajectory optimization.

== 1. ROBOT (Unitree Go2) ==

Mass: {mass:.2f} kg | Initial COM height: {initial_height:.4f}m
Body trunk: ~37.6cm long x 9.4cm wide x 11.4cm tall (model half-extents: 0.1881/0.0468/0.057m)
Leg length: ~43cm fully extended (21.3cm thigh + 21.3cm calf)
Hip spacing: Front/rear ~19cm from COM, Left/right ~14cm from COM
Joint limits: Hip ±46deg, Front thigh -90 to +92deg, Rear thigh -30 to +92deg, Calf -149 to -48deg
Per-component GRF limit: {grf_limit:.0f} N (fx,fy,fz each per foot) | Joint velocity limit: {jvel_limit:.1f} rad/s
Ground friction coefficient: {mu}

Physical capability limits (do NOT exceed):
- Max COM height gain: ~0.15-0.25m (normal jump), ~0.3m (aggressive)
- Max takeoff vz: ~1.8-2.5 m/s | Max flight duration: ~0.3-0.5s
- Max peak GRF: ~6-8x body weight (~900-1200 N total across 4 feet)
- Max COM acceleration: ~4-6g typical, up to ~13g solver-feasible
- Peak angular velocity: 8-15 rad/s
- vz_takeoff = g * flight_duration / 2, but flight_duration MUST be <= 0.5s

Key physics:
- Rotation: angle_change = angular_velocity × time
- Projectile motion: peak_height = initial_height + v²/(2g)
- Angular momentum is conserved during flight (no external torques)
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

Include these calls (items 4-6 are REQUIRED — solver FAILS without them):
  1. mpc.set_task_name("...")              # descriptive name (defaults to "unknown")
  2. mpc.set_duration(seconds)            # typically 1.0-2.0s (defaults to 1.0)
  3. mpc.set_time_step(0.02)              # defaults to 0.02
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
  Default weight for your constraints: 1e3. Physics constraints are always hard
  — you cannot soften them. Hard constraint names: friction_cone_constraints,
  foot_height_constraints, foot_velocity_constraints, joint_limits_constraints,
  input_limits_constraints, body_clearance_constraints, complementarity_constraints.
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
  k=horizon). The three system constraints (joint_limits, foot_height, body_clearance)
  are additionally applied at k=horizon by the solver, but this does NOT apply to
  any constraint you write. The initial state (k=0) is enforced separately by the
  solver. Design terminal constraints to tighten as progress approaches 1.0 — the
  last applied step is k=horizon-1 (progress = (horizon-1)/horizon, e.g. 0.98 for
  horizon=50).

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

P3 — DON'T OVER-CONSTRAIN: Use one-sided bounds (lower, cs.inf) or (-cs.inf, upper)
  when possible. Constrain RATES not positions for dynamic motions. Constraints define
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
  - body_clearance violations often mean too much rotation during ground contact
  - Use mpc.set_slack_weights() to prioritize which constraints matter most

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
  - Flight phases: ballistic z(t) = z0 + vz0*t - 0.5*g*t², angular momentum conserved, GRF=0
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

Python builtins: abs, round, max, min, len, range, enumerate, zip, float, int,
  str, list, tuple, dict, bool, hasattr, getattr, isinstance, type, print
Constants: pi, inf
Rotation: SO3 (from liecasadi)

== 7. ITERATION STRATEGY ==

ITERATION 1: Minimal viable constraints
  - Maximum 2-3 constraints, bounds 2-3x wider than you think necessary
  - Goal: Solver converges, motion happens (even if imperfect)

LATER ITERATIONS: You decide the strategy based on iteration history and metrics.
  - Solver failed → remove constraints, widen bounds, simplify
  - Solver converged but motion weak → tighten bounds, add constraints
  - Scores plateauing → structurally different approach (different phases, contact
    sequence, constraint variables, or rewrite from scratch)

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

⚠️ MANDATORY CHECKLIST - Items 4-6 are REQUIRED (solver FAILS without them):
□ mpc.set_task_name("...")              (recommended, defaults to "unknown")
□ mpc.set_duration(...)                 (recommended, defaults to 1.0)
□ mpc.set_time_step(0.02)              (recommended, defaults to 0.02)
□ mpc.set_contact_sequence(...)  ← REQUIRED - THIS IS THE #1 MISSING CALL
□ mpc.add_constraint(...)        ← REQUIRED
□ mpc.set_reference_trajectory(...)  ← REQUIRED - cost target AND initial guess

Other requirements:
- Constraint function must have 7 parameters: (x_k, u_k, kindyn_model, config, contact_k, k, horizon)
- Must return exactly 3 values: (constraint_expr, lower_bound, upper_bound)
- All return values must be CasADi MX expressions (use vertcat for multiple constraints)
- Constraints are applied at k=1 through k=horizon-1 (NOT at k=0). Bounds at k=1 must be reachable from the starting state (height={initial_height:.4f}m).

Return ONLY corrected Python code."""
