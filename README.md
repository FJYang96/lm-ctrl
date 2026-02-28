# LLM-Controlled Quadruped Locomotion

Natural language to dynamic quadruped behaviors. An LLM generates trajectory optimization constraints from commands like "do a backflip," a CasADi MPC solver plans the motion, and an RL tracking policy (OPT-Mimic) executes it in simulation.

## Architecture

```
"do a backflip"
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  LLM        │────▶│  CasADi MPC  │────▶│  RL Tracking  │
│  (Claude)   │     │  Optimizer   │     │  Policy (PPO) │
│             │     │              │     │               │
│ Generates   │     │ Plans state  │     │ Closed-loop   │
│ constraints │◀────│ trajectory,  │     │ execution in  │
│ from text   │ fb  │ GRFs, joints │     │ MuJoCo        │
└─────────────┘     └──────────────┘     └──────────────┘
```

**Three-stage pipeline:**
1. **LLM constraint generation** — Claude translates natural language into CasADi optimization constraints with iterative feedback refinement
2. **MPC trajectory optimization** — CasADi/IPOPT solves for state trajectories, ground reaction forces, and joint velocities using a Single Rigid Body dynamics model
3. **RL tracking policy** — A PPO policy (OPT-Mimic) tracks the planned trajectory in MuJoCo with PD control + J^T·F feedforward torques

## Quick Start

### LLM Pipeline
```bash
# Set up API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
pip install -r requirements_llm.txt

# Generate behaviors from natural language
python llm_main.py "do a backflip"
python llm_main.py "jump as high as possible"
```

### RL Tracking Policy
```bash
# Train and evaluate on an existing MPC trajectory
./rl/run_smoke_test.sh 3000000 8  # 3M steps, 8 parallel envs

# Or run steps individually:
python -m rl.train --state-traj results/state_traj.npy --grf-traj results/grf_traj.npy --joint-vel-traj results/joint_vel_traj.npy
python -m rl.evaluate --model-path rl/trained_models/smoke_test/best_model/best_model.zip
python -m rl.generate_frames  # extract frames for visual comparison
```

### MPC Only (No LLM)
```bash
./run_opti.sh
```

### Web Frontend
```bash
cd frontend && pip install -r requirements.txt && python app.py
```

## Project Structure

```
├── llm_integration/                    # LLM feedback pipeline
│   ├── pipeline/
│   │   ├── feedback_pipeline.py        # Main iterative loop: generate → optimize → simulate → score → feedback
│   │   ├── optimization.py             # Calls MPC solver, handles failures and retries
│   │   ├── simulation.py               # Runs MuJoCo simulation on planned trajectories, captures video
│   │   ├── constraint_generation.py    # Prompts LLM to generate CasADi constraint code
│   │   ├── feedback_context.py         # Builds context string (metrics, history, feedback) for LLM
│   │   └── utils.py                    # Iteration directory management, file I/O helpers
│   ├── feedback/
│   │   ├── llm_evaluation.py           # LLM-based scoring: success/failure判定, iteration summaries
│   │   ├── constraint_feedback.py      # Analyzes constraint design and suggests fixes
│   │   ├── reference_feedback.py       # RMSE metrics + physics plausibility of reference trajectory
│   │   ├── video_extraction.py         # Extracts key frames from videos as base64 for LLM vision
│   │   ├── format_metrics.py           # Formats numerical metrics for LLM prompts
│   │   └── format_hardness.py          # Constraint violation severity formatting
│   ├── client/
│   │   ├── llm_client.py               # Anthropic API wrapper with retry logic
│   │   └── code_extraction.py          # Parses Python code blocks from LLM responses
│   ├── executor/
│   │   ├── safe_executor.py            # Sandboxed exec() for LLM-generated constraint code
│   │   └── globals.py                  # Allowed functions/variables exposed to LLM code
│   ├── mpc/
│   │   ├── llm_task_mpc.py             # Wraps MPC solver with LLM-generated constraints
│   │   ├── constraint_wrapper.py       # Applies constraint code to CasADi Opti problem
│   │   ├── config_management.py        # MPC config overrides per iteration
│   │   └── contact_utils.py            # Contact sequence generation from phase descriptions
│   └── _prompts/
│       ├── system_prompt.py            # System prompt: robot specs, CasADi API, constraint format
│       └── user_prompts.py             # User prompt templates for each LLM call
│
├── mpc/                                # Core MPC solver
│   ├── mpc_opti.py                     # CasADi Opti-based MPC (base formulation)
│   ├── mpc_opti_slack.py               # Slack variable formulation for soft constraints
│   ├── mpc_config.py                   # MPC hyperparameters (horizon, dt, weights, bounds)
│   ├── constraints.py                  # Standard constraint definitions (friction, kinematics)
│   ├── config_complementarity.py       # Contact complementarity constraint config
│   └── dynamics/
│       └── model.py                    # Kinodynamic model: Jacobians, forward dynamics, mass matrix
│
├── rl/                                 # RL tracking policy (OPT-Mimic)
│   ├── tracking_env.py                 # Go2 Gymnasium env: 39D obs, PD+feedforward, multiplicative reward
│   ├── train.py                        # PPO training with SB3, LR schedule, domain randomization
│   ├── evaluate.py                     # Compute tracking errors (pos/ori/joint RMS) and save video
│   ├── rollout.py                      # Policy rollout — drop-in replacement for inverse dynamics
│   ├── feedforward.py                  # J^T·F feedforward torques via CasADi Jacobians
│   ├── reference.py                    # Wraps .npy arrays into per-timestep lookups (pos, vel, GRF, phase)
│   ├── callbacks.py                    # VecNormalize saver, training logger, reward plots, diagnostics
│   ├── generate_frames.py             # Extract video frames into folders for visual comparison
│   └── run_smoke_test.sh               # End-to-end: train → evaluate → generate comparison frames
│
├── utils/
│   ├── simulation.py                   # MuJoCo simulation loop: apply torques, capture frames
│   ├── inv_dyn.py                      # Open-loop inverse dynamics (J^T·F torque computation)
│   ├── visualization.py                # Trajectory comparison plots (planned vs simulated)
│   └── conversion.py                   # State format conversion between MPC and MuJoCo representations
│
├── frontend/
│   ├── app.py                          # Flask backend: API endpoints for running pipeline via web UI
│   └── templates/index.html            # Single-page frontend with live logs and video playback
│
├── main.py                             # Standalone MPC pipeline (no LLM): optimize → simulate → save
├── llm_main.py                         # LLM pipeline entry point: parse command → run feedback loop
├── config.py                           # Global config: robot, experiment, MPC parameters
└── configs/
    ├── experiments/base.py             # Experiment settings (sim_dt, render, friction)
    └── robots/robot_data.py            # Robot-specific parameters (Go2 mass, inertia, joint limits)
```

## Robot

**Unitree Go2** — 12-DOF quadruped (15kg). Simulated in MuJoCo via [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped).

## Key References

- **OPT-Mimic** (Fuchioka et al., 2023) — RL tracking policy architecture
- **CasADi** — Symbolic optimization framework for trajectory planning
- **MuJoCo** — Physics simulation for policy training and evaluation
- **Stable Baselines3** — PPO implementation

## Output Files

After running the pipeline:
- `results/state_traj.npy`, `grf_traj.npy`, `joint_vel_traj.npy` — planned trajectory
- `results/trajectory.mp4` — planned trajectory video (MuJoCo, open-loop)
- `results/rl_tracking.mp4` — RL tracking video (MuJoCo, closed-loop)
- `results/comparison/` — extracted frames for visual comparison
