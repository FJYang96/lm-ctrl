# RL Tracking Policy (OPT-Mimic)

Closed-loop tracking policy that replaces the open-loop inverse dynamics + simulation pipeline. Based on the [OPT-Mimic paper](https://arxiv.org/abs/2210.01247) (Fuchioka et al., 2023), adapted from Solo 8 (8 joints, 1.7kg) to Go2 (12 joints, 15kg).

## What This Replaces

The existing LLM pipeline executes MPC trajectories via inverse dynamics:

```
MPC trajectory
     |
     v
compute_joint_torques()          <-- utils/inv_dyn.py
  tau = M*q_ddot + C - J^T*F        (full inverse dynamics, open-loop)
     |
     v
simulate_trajectory()            <-- utils/simulation.py
  for each step:
    env.step(precomputed_torque)     (no feedback, errors compound)
     |
     v
qpos_traj, qvel_traj, grf_traj, images
```

This RL pipeline replaces both functions with a single closed-loop call:

```
MPC trajectory
     |
     v
execute_policy_rollout()         <-- rl/rollout.py
  for each step:
    obs = read actual robot state    (closed-loop feedback every 20ms)
    action = policy(obs)             (residual joint corrections)
    torque = PD(ref + action) + ff   (PD controller + J^T*F feedforward)
    env.step(torque)
     |
     v
qpos_traj, qvel_traj, grf_traj, images   (same output format)
```

## Pipeline Overview

```
                    TRAINING (once per trajectory)
                    ==============================

 MPC Output                     ReferenceTrajectory            Go2TrackingEnv
 (.npy files)        --->       (reference.py)        --->     (tracking_env.py)
 state_traj (N+1,30)            Phase-indexed API:             OPT-Mimic gym env:
 grf_traj (N,12)                - get_body_pos(phase)          - obs: 30D
 joint_vel_traj (N,12)          - get_joint_pos(phase)         - action: 12D residual
                                - get_joint_vel(phase)         - reward: 5 Gaussian terms
                                - get_phase_encoding(phase)    - PD + feedforward actuation
                     |
                     v
              FeedforwardComputer         PPO (train.py)
              (feedforward.py)            - 16 parallel envs
              tau_ff = J^T * F            - domain randomization
              precomputed (N,12)          - 2M timesteps
                                               |
                                               v
                                          trained policy (.zip)
                                          obs normalizer (.pkl)


                    DEPLOYMENT (per MPC solution)
                    =============================

 MPC Output  --->  execute_policy_rollout()  --->  qpos, qvel, grf, images
                   (rollout.py)
                   Loads trained policy,
                   runs closed-loop in MuJoCo
```

## Inputs and Outputs

### Pipeline Inputs

All inputs come from the MPC trajectory optimizer (same arrays used by inverse dynamics):

| Input | Shape | Description |
|---|---|---|
| `state_traj` | `(N+1, 30)` | MPC state trajectory: body pos (0:3), vel (3:6), euler (6:9), ang_vel (9:12), joint_pos (12:24), joint_vel (24:30) |
| `grf_traj` | `(N, 12)` | Ground reaction forces: 3 components per foot x 4 feet (FL, FR, RL, RR) |
| `joint_vel_traj` | `(N, 12)` | Joint velocities from MPC (12 joints) |
| `contact_sequence` | `(4, N)` | (optional) Binary contact flags per foot per timestep. Enables contact-consistency termination. |

Where `N` = number of MPC control steps (horizon), each lasting `control_dt = 0.02s` (50Hz).

### Pipeline Outputs

Identical format to `simulate_trajectory()`:

| Output | Shape | Description |
|---|---|---|
| `qpos_traj` | `(T, 19)` | MuJoCo joint positions: base pos (0:3), base quat (3:7), joints (7:19) |
| `qvel_traj` | `(T, 18)` | MuJoCo joint velocities: base lin_vel (0:3), base ang_vel (3:6), joints (6:18) |
| `grf_traj` | `(T, 12)` | Actual simulated contact forces (12D) |
| `images` | list of arrays | Rendered frames (if rendering enabled) |

Where `T` = number of simulation substeps = `N * (control_dt / sim_dt)`.

## File Descriptions

### `reference.py` — Reference Trajectory Wrapper

Wraps raw MPC numpy arrays into a clean phase-indexed API. Used by both training and deployment.

**Class: `ReferenceTrajectory`**

- Converts MPC Euler angles to quaternions (precomputed at init)
- Provides `get_body_pos(phase)`, `get_body_quat(phase)`, `get_joint_pos(phase)`, `get_joint_vel(phase)`, `get_body_vel(phase)`, `get_body_ang_vel(phase)` for any phase index
- `get_phase_encoding(phase)` returns `[cos(2*pi*phase/N), sin(2*pi*phase/N)]` (sinusoidal encoding, OPT-Mimic)
- `get_feedforward_torque(phase)` returns precomputed `J^T*F` for the PD controller
- `get_contact_state(phase)` and `is_near_contact_transition(phase, foot, window)` support contact-consistency termination
- Clamped access: phases past the end return the last valid value
- `from_files()` class method loads directly from `.npy` files

### `feedforward.py` — Feedforward Torque Computer

Computes `tau_ff = J^T * F` (Jacobian-transpose times GRF) for all 4 feet. This is directly from OPT-Mimic Section III-B: "τ(φ) can be obtained through Jacobian transpose τ = J^T(−f)". Used as the feedforward term τ̂(φ) in Eq. 14.

**Class: `FeedforwardComputer`**

- Builds a CasADi symbolic function at init: `(base_pos, base_rpy, joint_pos, grf) -> tau_ff (12D)`
- Uses the kinodynamic model's leg Jacobians (FL, FR, RL, RR feet)
- `compute(base_pos, base_rpy, joint_pos, grf)` — single timestep
- `precompute_trajectory(ref)` — evaluates for all `N` timesteps, returns `(N, 12)` array

This is a **subset** of what inverse dynamics computes. Inverse dynamics does `tau = M*q_ddot + C - J^T*F` (full equation of motion). The feedforward computer only does `J^T*F` (the GRF contribution). The RL policy + PD controller implicitly handle the `M*q_ddot + C` part through closed-loop feedback.

### `tracking_env.py` — OPT-Mimic Gym Environment

The core training environment. Implements the full OPT-Mimic framework adapted for Go2.

**Class: `Go2TrackingEnv`** (gymnasium.Env)

**Observation (30D):**
- Body quaternion (4D) — from MuJoCo
- Joint positions (12D) + domain-randomized offset
- Joint velocities (12D)
- Phase encoding (2D) — sinusoidal `[cos, sin]`

Note: no base position or velocity in obs (proprioceptive only, matching paper).

**Action (12D):**
- Residual joint position corrections, scaled from `[-1, 1]` to `[-ACTION_LIMIT, +ACTION_LIMIT]` where `ACTION_LIMIT = 2.7 rad`

**Actuation (OPT-Mimic Eq. 14):**
```
target_pos = ref_joint_pos + action + joint_offset
torque = Kp * (target_pos - actual_pos)
       + Kd * (ref_vel - actual_vel)
       + feedforward_torque
torque = clip(torque, -33.5, +33.5) * torque_scale
```

PD gains scaled from paper (Kp=3, Kd=0.3 for 1.7kg Solo 8) by ~9x mass ratio: `Kp=25, Kd=1.5` for 15kg Go2.

**Reward (5 Gaussian terms, OPT-Mimic Eq. 15-16, Table I):**

| Term | Weight | Sigma | What it tracks |
|---|---|---|---|
| Position | 0.3 | 0.05 | Body COM position error |
| Orientation | 0.3 | 0.14 | Quaternion error (vector part) |
| Joint | 0.2 | 0.3 | Joint angle error |
| Smoothness | 0.1 | 0.35 | Action rate (consecutive action difference) |
| Torque | 0.1 | 3.0 | Max absolute joint torque |

Each term: `r_x = exp(-||error||^2 / (2 * sigma^2))`. Total reward in `[0, 1]`.

**Termination:**
1. Any tracking error exceeds `2.5 * sigma` for its reward component
2. Body height drops below 5cm (fall detection)
3. Foot contact state doesn't match reference (with 120ms / 6-step grace window near transitions)

**Domain Randomization (OPT-Mimic Table I):**

| Parameter | Distribution | Purpose |
|---|---|---|
| Ground friction | Uniform [0.55, 1.05] | Surface variation |
| Ground restitution | N(0.0, 0.25), clipped [0, 1] | Bounce variation |
| Joint position offset | N(0.0, 0.02) per joint | Encoder calibration error |
| Torque scale | N(1.0, 0.1), clipped [0.5, 1.5] | Motor/control mismatch |

**Reference State Initialization:** each episode starts at a random phase along the trajectory, so the policy learns all phases in parallel (critical for dynamic motions like flips).

**Timing:** policy runs at 50Hz (`control_dt = 0.02s`), physics at 1kHz (`sim_dt = 0.001s`), so 20 substeps per policy step.

### `train.py` — Training Script

Orchestrates the full training pipeline.

**Flow:**
1. `build_reference()` — loads `.npy` files into `ReferenceTrajectory`, precomputes feedforward torques via `FeedforwardComputer`
2. Creates 16 parallel training envs (`SubprocVecEnv`) with domain randomization enabled
3. Creates 1 eval env without randomization
4. Wraps envs in `VecNormalize` (obs normalization, no reward normalization)
5. Configures PPO with OPT-Mimic hyperparameters:
   - LR: `1e-3 * 0.99^update` (exponential decay)
   - Batch: 5000, buffer: 20000 samples/update, 40 epochs
   - Policy net: `[128, 128]`, value net: `[512, 512]`, ReLU, orthogonal init
   - Gamma: 0.995, GAE lambda: 0.95, clip: 0.2, no entropy bonus
6. Trains for 2M timesteps with checkpoint + eval callbacks
7. Saves final model, best model, and `VecNormalize` stats

**Usage:**
```bash
python -m rl.train \
    --state-traj results/state_traj.npy \
    --grf-traj results/grf_traj.npy \
    --joint-vel-traj results/joint_vel_traj.npy \
    --output-dir rl/trained_models \
    --total-timesteps 2000000 \
    --num-envs 16
```

**Outputs:**
```
rl/trained_models/
  tracking_policy_final.zip     # final PPO model
  vec_normalize.pkl             # observation normalization stats
  best_model/
    best_model.zip              # best model by eval reward
  checkpoints/
    tracking_policy_50000_steps.zip
    tracking_policy_100000_steps.zip
    ...
  tb_logs/                      # TensorBoard logs
```

### `rollout.py` — Policy Rollout (Drop-in Replacement)

The deployment entry point. Replaces `compute_joint_torques() + simulate_trajectory()` with a single function call.

**Function: `execute_policy_rollout(env, state_traj, grf_traj, joint_vel_traj, kindyn_model, policy, ...)`**

**Flow:**
1. Builds `ReferenceTrajectory` from the MPC arrays
2. Precomputes feedforward torques via `FeedforwardComputer`
3. Loads `VecNormalize` stats for observation normalization (if available)
4. Resets MuJoCo env to reference initial state
5. For each phase (0 to N-1):
   - Reads actual robot state from MuJoCo (closed-loop)
   - Builds 30D observation (same as training env)
   - Normalizes observation using saved stats
   - Queries policy for action (deterministic)
   - Computes PD + feedforward torque (same as training env)
   - Steps MuJoCo for `substeps` iterations
   - Records qpos, qvel, grf, and optionally renders frames
6. Returns `(qpos_traj, qvel_traj, grf_traj, images)` — same format as `simulate_trajectory()`

**Integration with LLM pipeline:**
```python
# Before (llm_integration/pipeline/simulation.py):
joint_torques = compute_joint_torques(kindyn, state_traj, grf_traj, contact_seq, dt, joint_vel)
qpos, qvel, grf, images = simulate_trajectory(env, joint_torques, planned_images)

# After:
from rl.rollout import execute_policy_rollout
qpos, qvel, grf, images = execute_policy_rollout(
    env, state_traj, grf_traj, joint_vel_traj, kindyn, policy
)
```

### `evaluate.py` — Standalone Evaluation

Loads a trained policy, runs a rollout, computes tracking errors, and saves a video.

**Usage:**
```bash
python -m rl.evaluate \
    --model-path rl/trained_models/best_model/best_model.zip \
    --state-traj results/state_traj.npy \
    --grf-traj results/grf_traj.npy \
    --joint-vel-traj results/joint_vel_traj.npy \
    --output-video results/rl_tracking.mp4
```

**Output:** prints RMS tracking errors (position, orientation, joint angles) and saves a video of the rollout.

## Why RL Over Inverse Dynamics

| | Inverse Dynamics (open-loop) | RL Tracking (closed-loop) |
|---|---|---|
| Feedback | None. Torques precomputed from plan. | Reads actual state every 20ms, corrects. |
| Model mismatch | SRB assumes massless legs, perfect contacts. Full-model sim diverges. | Policy trained on full model with domain randomization. |
| Contact impacts | Finite-diff accelerations are noisy at landings. Torques spike or are wrong. | No acceleration computation needed. PD + learned residuals handle impacts. |
| Error accumulation | Small errors compound — no correction mechanism. | Self-correcting — errors don't accumulate. |
| Robustness | Fragile. Works only if sim matches plan exactly. | Robust to friction, restitution, calibration, and torque variations. |

## OPT-Mimic Correspondence

Every component maps directly to the paper:

| Our Code | Paper Reference |
|---|---|
| `feedforward.py`: `J^T * F` | Section III-B: "τ(φ) = J^T(−f)" |
| PD + residual + feedforward actuation | Eq. 14, Fig. 3 ("Pos+Vel+Torque" config) |
| Obs: quat + joints + jvel + phase | Section III-C.1 |
| 5 Gaussian reward terms | Eq. 15-16, Table I |
| Weights [0.3, 0.3, 0.2, 0.1, 0.1] | Table I (exact match) |
| Sigmas [0.05, 0.14, 0.3, 0.35, 3.0] | Table I (exact match) |
| 2.5*sigma termination | Section III-C.4 |
| Contact consistency + 120ms grace | Section III-C.4, condition 3 |
| Random phase initialization | Section III-C.4 |
| Domain rand: friction, restitution, joint offset, torque scale | Table I (exact match) |
| 50Hz policy, 1kHz sim | Section III-C.2 |
| PPO | Section III-C |
| PD gains Kp=25, Kd=1.5 | Paper Kp=3, Kd=0.3 for 1.7kg Solo 8, scaled ~9x for 15kg Go2 |
