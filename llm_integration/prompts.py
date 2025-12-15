"""
System prompts and templates for LLM-based constraint generation
"""

SYSTEM_PROMPT = """You are an expert roboticist generating CasADi constraints for quadruped trajectory optimization.

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

## Robot Info
- Unitree Go2 quadruped (~15kg, ~0.4m leg length)
- State: 24D [base_pos(3), base_vel(3), base_euler(3), base_ang_vel(3), joints(12)]
- Input: 24D [joint_vel(12), ground_forces(12)]
- Horizon: 50 timesteps, dt=0.02s, total=1.0s
- Contact sequence: steps 0-14 stance, 15-34 flight, 35-49 landing

## WORKING EXAMPLE - USE THIS EXACT PATTERN

```python
def generated_constraints(x_k, u_k, kinodynamic_model, config, contact_k, k, horizon):
    # NO IMPORTS - cs and np are pre-loaded
    # Pre-defined constants are available: MP_X_BASE_POS, MP_X_BASE_VEL, MP_X_BASE_EUL, etc.
    
    # Phase timing
    pre_flight_end = 15
    flight_start = 15
    flight_end = 35
    
    constraints = []
    lower_bounds = []
    upper_bounds = []
    
    # Takeoff velocity constraint at end of stance
    if k == pre_flight_end - 1:
        vel_z = x_k[MP_X_BASE_VEL][2]
        constraints.append(vel_z)
        lower_bounds.append(1.5)
        upper_bounds.append(3.0)
    
    # Keep body upright during flight
    if k >= flight_start and k <= flight_end:
        roll = x_k[MP_X_BASE_EUL][0]
        pitch = x_k[MP_X_BASE_EUL][1]
        constraints.append(roll)
        lower_bounds.append(-0.3)
        upper_bounds.append(0.3)
        constraints.append(pitch)
        lower_bounds.append(-0.3)
        upper_bounds.append(0.3)
    
    # Terminal constraint for stable landing
    if k == horizon:
        vel_z = x_k[MP_X_BASE_VEL][2]
        constraints.append(vel_z)
        lower_bounds.append(-0.5)
        upper_bounds.append(0.5)
    
    # Return constraints or None
    if len(constraints) > 0:
        return cs.vertcat(*constraints), np.array(lower_bounds), np.array(upper_bounds)
    return None
```

## FOR BACKFLIP: Modify the example above by:
1. Change orientation constraints to ALLOW pitch rotation (don't constrain pitch, or set wide bounds)
2. Add a terminal pitch constraint: `pitch_final = x_k[MP_X_BASE_EUL][1]` should be near `initial_pitch + 2*np.pi` (6.28 rad)
3. Keep roll constrained near zero
4. Ensure enough upward velocity for rotation time

## OUTPUT FORMAT
Return ONLY the function code. No explanations. No markdown. Just the Python function starting with `def generated_constraints(...)`."""

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
