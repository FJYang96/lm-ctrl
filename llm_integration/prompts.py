"""
System prompts and templates for LLM-based constraint generation
"""

from typing import Any, Optional

import numpy as np

# Base system prompt with constraint rules (static parts)
SYSTEM_PROMPT_BASE = """You are an expert roboticist generating CasADi constraints for quadruped trajectory optimization.

## CRITICAL RULES - READ CAREFULLY

1. **NO IMPORTS** - `cs` (casadi) and `np` (numpy) are pre-loaded. Do NOT write `import` statements.
2. **Use pre-defined constants** - These are already available:
   - `MP_X_BASE_POS = slice(0, 3)` for base position [x, y, z]
   - `MP_X_BASE_VEL = slice(3, 6)` for base velocity [vx, vy, vz]
   - `MP_X_BASE_EUL = slice(6, 9)` for base orientation [roll, pitch, yaw]
   - `MP_X_BASE_ANG = slice(9, 12)` for base angular velocity
   - `MP_X_Q = slice(12, 24)` for joint positions
3. **Return format**: Return `(cs.vertcat(*list), np.array(list), np.array(list))` or `None`
4. **Use simple Python** - Only use: `if`, `elif`, `else`, `for`, `range`, `len`, `int`, `float`, `list`, `append`

## EXISTING MPC CONSTRAINTS (ALREADY ENFORCED BY THE SOLVER)

Your generated constraints will be ADDED to these existing constraints. Do NOT duplicate them.
Design your constraints to be COMPATIBLE with these physical limits:

### 1. Friction Cone Constraints
- For each foot (4 total), ground reaction forces must satisfy Coulomb friction:
  - Normal force f_z: 0 <= f_z <= {grf_limit}N (when in contact), f_z ≈ 0 (when in swing)
  - Tangential forces bounded by friction: |f_x| <= μ * f_z and |f_y| <= μ * f_z
  - Friction coefficient μ ≈ {friction_coef} (typical ground)
- **Implication**: Total vertical force limited to {total_grf_limit}N (4 feet × {grf_limit}N)

### 2. Foot Height Constraints
- Stance feet: foot height constrained to ≈0 (contact with ground, tolerance ~0.02m)
- Swing feet: foot height >= 0 (must stay above ground)
- **Implication**: During stance phases, feet are fixed to ground

### 3. Foot Velocity (No-Slip) Constraints
- Stance feet: foot velocity ≈ 0 in all directions (tolerance ~0.005 m/s)
- **Implication**: No sliding during contact, feet act as fixed pivots

### 4. Joint Limits Constraints
- Hip abduction/adduction: [-0.863, 0.863] rad (±49°)
- Hip flexion/extension: [-0.686, 3.48] rad
- Knee flexion: [-2.82, -0.888] rad (always bent)
- **Implication**: Joint positions must stay within these ranges throughout motion

### 5. Input Limits Constraints
- Joint velocities: [-{joint_vel_limit}, {joint_vel_limit}] rad/s for all 12 joints
- Ground reaction forces: [-{grf_limit}, {grf_limit}] N per axis per foot
- **Implication**: Limited actuation speed and force generation

### 6. Body Clearance Constraints
- COM height must maintain clearance considering body tilt:
  - effective_clearance = z_com - (0.05 + 0.15*|pitch| + 0.10*|roll|) >= 0.01m
- **Implication**: Large pitch/roll angles require higher COM to avoid ground collision

### 7. Complementarity Constraints
- f_z * v_z <= ε (force and velocity can't both be large)
- **Implication**: Smooth contact transitions, no impact forces

## FEASIBILITY GUIDELINES

To generate constraints that the solver CAN satisfy:

1. **Force limits**: With ~{robot_mass}kg robot and max {total_grf_limit}N total vertical force,
   max vertical acceleration ≈ {max_accel:.1f} m/s².

2. **Velocity achievability**: Consider stance duration when setting velocity targets.
   Longer stance = more time to accelerate = higher achievable velocities.

3. **Use SOFT bounds**: Instead of exact equality constraints, use inequality ranges:
   - BAD: `constraint == target` (hard to satisfy exactly)
   - GOOD: `lower_bound = target - tolerance, upper_bound = target + tolerance`

4. **Gradual transitions**: Avoid discontinuous constraint jumps between timesteps.

5. **Phase-aware constraints**: Use the `contact_k` parameter to know which feet are in contact.
   - contact_k is a 4-element array [FL, FR, RL, RR], where 1=stance, 0=swing
   - Use `k` (current timestep) and `horizon` to determine motion phase

## MOTION-SPECIFIC TIPS

- **Jumping**: Constrain upward velocity at takeoff (1.5-3.0 m/s realistic), minimize horizontal drift
- **Flips/Rotations**: Allow orientation freedom during flight, constrain terminal orientation
- **Walking/Trotting**: Maintain body height, alternate leg contacts, forward velocity
- **Spinning**: Constrain yaw rate, maintain height, keep body level (roll/pitch near 0)
- **Landing**: Constrain terminal velocity to be small, ensure stable final pose
"""

# Dynamic robot and trajectory info template
ROBOT_INFO_TEMPLATE = """
## Robot Info
- {robot_name} quadruped (~{robot_mass}kg)
- State: 24D [base_pos(3), base_vel(3), base_euler(3), base_ang_vel(3), joints(12)]
- Input: 24D [joint_vel(12), ground_forces(12)]
- Horizon: {horizon} timesteps, dt={dt}s, total={duration}s
"""

# Contact sequence info template
CONTACT_SEQUENCE_TEMPLATE = """
## Contact Sequence
{contact_description}
- Use `contact_k[i]` to check if foot i is in stance (1) or swing (0)
- Foot indices: 0=FL (front-left), 1=FR (front-right), 2=RL (rear-left), 3=RR (rear-right)
"""

# Code example template
CODE_EXAMPLE = """
## WORKING EXAMPLE - USE THIS EXACT PATTERN

```python
def generated_constraints(x_k, u_k, kinodynamic_model, config, contact_k, k, horizon):
    # NO IMPORTS - cs and np are pre-loaded
    # Pre-defined constants are available: MP_X_BASE_POS, MP_X_BASE_VEL, MP_X_BASE_EUL, etc.

    constraints = []
    lower_bounds = []
    upper_bounds = []

    # Example: Constrain base height throughout motion
    base_height = x_k[MP_X_BASE_POS][2]
    constraints.append(base_height)
    lower_bounds.append(0.15)  # Min height
    upper_bounds.append(0.8)   # Max height

    # Example: Terminal constraint for stable landing
    if k == horizon:
        # Small vertical velocity at end
        vel_z = x_k[MP_X_BASE_VEL][2]
        constraints.append(vel_z)
        lower_bounds.append(-0.5)
        upper_bounds.append(0.5)

        # Level body at end
        roll = x_k[MP_X_BASE_EUL][0]
        pitch = x_k[MP_X_BASE_EUL][1]
        constraints.append(roll)
        lower_bounds.append(-0.2)
        upper_bounds.append(0.2)
        constraints.append(pitch)
        lower_bounds.append(-0.2)
        upper_bounds.append(0.2)

    # Return constraints or None
    if len(constraints) > 0:
        return cs.vertcat(*constraints), np.array(lower_bounds), np.array(upper_bounds)
    return None
```

## OUTPUT FORMAT
Return ONLY the function code. No explanations. No markdown. Just the Python function starting with `def generated_constraints(...)`."""


def build_system_prompt(
    config: Any,
    contact_sequence: Optional[np.ndarray] = None,
) -> str:
    """
    Build a complete system prompt with dynamic robot/trajectory information.

    Args:
        config: Configuration object containing robot_data, mpc_config, experiment
        contact_sequence: Optional contact sequence array (4 x horizon)

    Returns:
        Complete system prompt string
    """
    # Extract robot parameters
    robot_mass = getattr(config.robot_data, "mass", 15.0)
    robot_name = getattr(config.robot_data, "name", "Unitree Go2")
    grf_limit = getattr(config.robot_data, "grf_limits", 200.0)
    joint_vel_limit = getattr(config.robot_data, "joint_velocity_limits", 21.0)
    if hasattr(joint_vel_limit, "__iter__"):
        joint_vel_limit = float(np.max(joint_vel_limit))

    # Extract MPC parameters
    dt = getattr(config.mpc_config, "mpc_dt", 0.02)
    duration = getattr(config.mpc_config, "duration", 1.0)
    horizon = int(duration / dt)

    # Extract experiment parameters
    friction_coef = getattr(config.experiment, "mu_ground", 0.6)

    # Calculate derived values
    total_grf_limit = 4 * grf_limit
    gravity = 9.81
    max_accel = (total_grf_limit / robot_mass) - gravity

    # Build base prompt with filled parameters
    base_prompt = SYSTEM_PROMPT_BASE.format(
        grf_limit=grf_limit,
        friction_coef=friction_coef,
        total_grf_limit=total_grf_limit,
        joint_vel_limit=joint_vel_limit,
        robot_mass=robot_mass,
        max_accel=max_accel,
    )

    # Build robot info section
    robot_info = ROBOT_INFO_TEMPLATE.format(
        robot_name=robot_name,
        robot_mass=robot_mass,
        horizon=horizon,
        dt=dt,
        duration=duration,
    )

    # Build contact sequence description
    contact_description = _describe_contact_sequence(contact_sequence, dt)
    contact_info = CONTACT_SEQUENCE_TEMPLATE.format(
        contact_description=contact_description,
    )

    # Combine all parts
    full_prompt = base_prompt + robot_info + contact_info + CODE_EXAMPLE

    return full_prompt


def _describe_contact_sequence(
    contact_sequence: Optional[np.ndarray],
    dt: float,
) -> str:
    """
    Generate a human-readable description of the contact sequence.

    Args:
        contact_sequence: Array of shape (4, horizon) with 1=stance, 0=swing
        dt: Timestep duration

    Returns:
        Description string
    """
    if contact_sequence is None:
        return "- Contact sequence not specified. Use `contact_k` parameter to check stance/swing."

    # Analyze the contact pattern
    # Check if all feet have same pattern (synchronized)
    all_same = all(
        np.array_equal(contact_sequence[0], contact_sequence[i]) for i in range(1, 4)
    )

    if all_same:
        # Synchronized gait - describe phases
        pattern = contact_sequence[0]
        phases = _find_phases(pattern)

        lines = ["- All feet synchronized (same contact pattern):"]
        for phase_type, start, end in phases:
            start_time = start * dt
            end_time = end * dt
            phase_name = (
                "STANCE (all feet on ground)"
                if phase_type == 1
                else "FLIGHT (all feet in air)"
            )
            lines.append(
                f"  - Steps {start}-{end} ({start_time:.2f}s - {end_time:.2f}s): {phase_name}"
            )

        return "\n".join(lines)
    else:
        # Alternating gait (walking, trotting, etc.)
        lines = ["- Alternating gait pattern (feet have different contact timings):"]

        foot_names = [
            "FL (front-left)",
            "FR (front-right)",
            "RL (rear-left)",
            "RR (rear-right)",
        ]
        for i, name in enumerate(foot_names):
            pattern = contact_sequence[i]
            stance_ratio = np.mean(pattern) * 100
            lines.append(f"  - {name}: {stance_ratio:.0f}% stance phase")

        # Check for common patterns
        fl_fr_same = np.array_equal(contact_sequence[0], contact_sequence[1])
        rl_rr_same = np.array_equal(contact_sequence[2], contact_sequence[3])
        fl_rr_same = np.array_equal(contact_sequence[0], contact_sequence[3])

        if fl_rr_same and np.array_equal(contact_sequence[1], contact_sequence[2]):
            lines.append("  - Pattern: TROT gait (diagonal pairs move together)")
        elif fl_fr_same and rl_rr_same:
            lines.append("  - Pattern: BOUND gait (front pair, then rear pair)")

        return "\n".join(lines)


def _find_phases(pattern: np.ndarray) -> list:
    """Find contiguous phases in a binary pattern."""
    phases = []
    if len(pattern) == 0:
        return phases

    current_type = pattern[0]
    start = 0

    for i in range(1, len(pattern)):
        if pattern[i] != current_type:
            phases.append((current_type, start, i - 1))
            current_type = pattern[i]
            start = i

    phases.append((current_type, start, len(pattern) - 1))
    return phases


# Keep SYSTEM_PROMPT for backwards compatibility (will be overridden at runtime)
SYSTEM_PROMPT = (
    SYSTEM_PROMPT_BASE.format(
        grf_limit=200,
        friction_coef=0.6,
        total_grf_limit=800,
        joint_vel_limit=21,
        robot_mass=15,
        max_accel=43.5,
    )
    + ROBOT_INFO_TEMPLATE.format(
        robot_name="Unitree Go2",
        robot_mass=15,
        horizon=50,
        dt=0.02,
        duration=1.0,
    )
    + CONTACT_SEQUENCE_TEMPLATE.format(
        contact_description="- Contact sequence will be provided at runtime via `contact_k` parameter",
    )
    + CODE_EXAMPLE
)

FEEDBACK_PROMPT_TEMPLATE = """
## Previous Iteration Results

**Command**: {command}

**Generated Constraints**:
```python
{previous_constraints}
```

**Optimization Status**: {optimization_status}
- Solver converged: {converged}
- Constraint violations: {constraint_violations}
- Maximum constraint violation: {max_violation}

**Trajectory Analysis**:
- Initial height: {initial_height:.3f}m
- Maximum height: {max_height:.3f}m
- Final height: {final_height:.3f}m
- Initial pitch: {initial_pitch:.3f} rad
- Final pitch: {final_pitch:.3f} rad
- Pitch change: {pitch_change:.3f} rad (target: 2π = {target_pitch:.3f} rad)

**Success Metrics**:
- Height clearance achieved: {height_clearance_ok}
- Rotation target achieved: {rotation_target_ok}
- Landing stability: {landing_stable}
- Overall success: {overall_success}

**Issues Identified**:
{issues_identified}

## Feedback for Next Iteration

Based on these results, please modify your constraints to address the identified issues. Key areas for improvement:

{improvement_suggestions}

Remember to:
1. Keep physically realizable constraints
2. Adjust bounds rather than completely changing constraint structure
3. Consider phased approach (different constraints for takeoff/flight/landing)
4. Maintain numerical stability

Generate an improved constraint function that addresses these specific issues.
"""

COMMAND_EXAMPLES = {
    "backflip": {
        "description": "Execute a backward somersault with full 2π rotation",
        "key_constraints": [
            "Initial/terminal stable stance",
            "Full backward rotation (pitch += 2π)",
            "Ground clearance during flight",
            "Controlled takeoff velocity",
            "Stable landing",
        ],
    },
    "jump": {
        "description": "Vertical jump with maximum height",
        "key_constraints": [
            "Vertical takeoff velocity",
            "Maximum height achievement",
            "Controlled landing",
            "Minimal horizontal drift",
        ],
    },
    "front_flip": {
        "description": "Forward somersault with full rotation",
        "key_constraints": [
            "Forward rotation (pitch -= 2π)",
            "Ground clearance",
            "Forward momentum control",
        ],
    },
}
