# Quadruped Hopping MPC - Acados & CasADi Opti

This repository contains both the original Acados-based and the new CasADi Opti-based implementations of a quadruped hopping MPC controller.

## Quick Start

### Run the Opti Version (Recommended)
```bash
./run_opti.sh
```

### Run the Original Acados Version
```bash
./run_acados.sh
```

### Manual Docker Run (if needed)
```bash
docker run --rm -v $(pwd):/home/lm-ctrl --workdir /home/lm-ctrl quadruped-lm-ctrl \
  /opt/conda/bin/conda run -n quadruped_pympc_ros2_env python main.py --solver opti
```

## Output Files

Both versions generate:
- **Trajectory data**: `state_traj*.npy`, `grf_traj*.npy`, `joint_vel_traj*.npy`
- **Videos**: `planned_traj*.mp4`, `trajectory*.mp4`
- Opti version files have `_opti` suffix

## Key Files

### Core Implementation
- `examples/mpc_opti.py` - New CasADi Opti-based MPC
- `examples/mpc.py` - Original Acados-based MPC
- `examples/model.py` - Kinodynamic model (shared)
- `config.py` - Robot and MPC parameters

### Main Scripts
- `main.py` - Unified pipeline (use `--solver opti` or `--solver acados`)

### Docker Scripts
- `run_opti.sh` - Run Opti version in Docker
- `run_acados.sh` - Run original version in Docker

## CasADi Opti Advantages

1. **Intuitive Formulation** - Direct mathematical constraint specification
2. **Faster Development** - No C code compilation overhead
3. **Better Debugging** - Clear symbolic expressions
4. **Future Ready** - Supports complementarity constraints
5. **Equivalent Results** - Same quality optimization as Acados

## Technical Details

### Opti vs Acados Architecture Comparison

**Original Acados Implementation:**
```python
# Complex setup with separate bound arrays
ocp.constraints.lh = np.concatenate((lb_friction, lb_height, lb_velocity))
ocp.constraints.uh = np.concatenate((ub_friction, ub_height, ub_velocity))

# Compiled C code generation
acados_ocp_solver = AcadosOcpSolver(ocp, build=True, generate=True)
```

**New Opti Implementation:**
```python
# Decision variables
self.X = self.opti.variable(self.states_dim, self.horizon + 1)
self.U = self.opti.variable(self.inputs_dim, self.horizon)

# Objective function
self.opti.minimize(cost)

# Direct mathematical constraints
self.opti.subject_to(x_next == x_k + dt * f_k)  # Dynamics
self.opti.subject_to(f_tangential_norm <= self.P_mu * f[2])  # Friction
self.opti.subject_to(height >= 0)  # Foot height

# Direct solving
sol = self.opti.solve()
```

### Implementation Features

**Cost Function:**
- Quadratic tracking cost for states and inputs
- Separate terminal cost for final state
- Weights from config parameters preserved

**Constraints:**
1. **Dynamics Constraints** - Using existing `forward_dynamics` from kinodynamic model
2. **Friction Cone** - Coulomb friction with contact-dependent activation
3. **Foot Height** - Ground clearance constraints based on contact sequence
4. **Foot Velocity** - Velocity limits for contacted feet
5. **Input Bounds** - Joint velocity and force limits

**Solver Configuration:**
- IPOPT solver with appropriate tolerances
- Smart initial guess with gravity compensation
- Error handling for failed optimizations

### Parameter Management
```python
# Runtime parameters
self.P_contact = self.opti.parameter(4, self.horizon)
self.P_mu = self.opti.parameter()
self.P_grf_min = self.opti.parameter()

# Set at solve time
self.opti.set_value(self.P_contact, contact_sequence)
```

## Future Enhancements

The Opti framework enables several future improvements:

1. **Complementarity Constraints** - For automatic contact detection
2. **Variable Contact Sequences** - Optimization over gait patterns  
3. **Multi-Contact Scenarios** - Complex terrain interactions
4. **Real-time Capabilities** - With warm-starting and reduced horizons

## Requirements

- Docker (for containerized execution)
- All dependencies are included in the Docker image

This transcription provides a solid foundation for advanced trajectory optimization research while maintaining compatibility with the existing codebase.