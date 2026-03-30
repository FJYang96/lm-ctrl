#!/bin/bash
# PD + Feedforward baseline (no RL residuals).
# Shows tracking quality without any learned policy — just the PD controller + J^T*F.
#
# Usage:
#   docker run --gpus all -v $(pwd):/workspace/lm-ctrl --entrypoint bash lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/run_baseline.sh
# docker run --gpus all -v $(pwd):/workspace/lm-ctrl --entrypoint bash lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/run_baseline.sh
set -e

# Clean previous baseline outputs
rm -rf rl_isaac/eval_output/* 2>/dev/null
echo "Cleaned rl_isaac/eval_output/"

GPU_ID=${GPU_ID:-2}
export CUDA_VISIBLE_DEVICES=$GPU_ID

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Lab Python not found. Run inside lm-ctrl-isaaclab Docker."
    exit 1
fi

ITER_DIR="results/llm_iterations/backflip"
STATE_TRAJ=${1:-$ITER_DIR/state_traj_iter_20.npy}
GRF_TRAJ=${2:-$ITER_DIR/grf_traj_iter_20.npy}
JOINT_VEL_TRAJ=${3:-$ITER_DIR/joint_vel_traj_iter_20.npy}
CONTACT_SEQ=${4:-$ITER_DIR/contact_sequence_iter_20.npy}

OUTPUT_DIR="rl_isaac/eval_output/baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"
export MUJOCO_GL=egl

CONTACT_ARG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_ARG="contact_sequence_path='$CONTACT_SEQ'"
fi

echo "Running PD+FF baseline (zero RL residuals)..."

$ISAAC_PYTHON -c "
import os, sys, types
os.environ['MUJOCO_GL'] = 'egl'
if 'mujoco.viewer' not in sys.modules:
    _fv = types.ModuleType('mujoco.viewer')
    _fv.Handle = type('Handle', (), {})
    sys.modules['mujoco.viewer'] = _fv
if 'glfw' not in sys.modules:
    _fg = types.ModuleType('glfw')
    _fg._glfw = True
    sys.modules['glfw'] = _fg
    sys.modules['glfw.library'] = types.ModuleType('glfw.library')

import mujoco
import numpy as np
import go2_config
from mpc.dynamics.model import KinoDynamic_Model
from rl_isaac.feedforward import FeedforwardComputer
from rl_isaac.reference import ReferenceTrajectory
from utils.conversion import sim_to_mpc

KP, KD = 25.0, 1.5
TORQUE_LIMITS = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float32)

state_traj = np.load('$STATE_TRAJ')
grf_traj = np.load('$GRF_TRAJ')
joint_vel_traj = np.load('$JOINT_VEL_TRAJ')
contact_seq = np.load('$CONTACT_SEQ') if os.path.exists('$CONTACT_SEQ') else None

control_dt = go2_config.mpc_config.mpc_dt
sim_dt = 0.001
substeps = int(control_dt / sim_dt)

kindyn = KinoDynamic_Model()
ref = ReferenceTrajectory(
    state_traj=state_traj, joint_vel_traj=joint_vel_traj,
    grf_traj=grf_traj, contact_sequence=contact_seq, control_dt=control_dt,
)
ff = FeedforwardComputer(kindyn)
ref.set_feedforward(ff.precompute_trajectory(ref))

from gym_quadruped.quadruped_env import QuadrupedEnv
quad_env = QuadrupedEnv(
    robot='go2', scene='flat', ground_friction_coeff=0.8,
    state_obs_names=QuadrupedEnv._DEFAULT_OBS + ('contact_forces:base',),
    sim_dt=sim_dt,
)

init_qpos = np.zeros(19)
init_qpos[0:3] = ref.get_body_pos(0)
init_qpos[3:7] = ref.get_body_quat(0)
init_qpos[7:19] = ref.get_joint_pos(0)
init_qvel = np.zeros(18)
init_qvel[0:3] = ref.get_body_vel(0)
init_qvel[3:6] = ref.get_body_ang_vel(0)
init_qvel[6:18] = ref.get_joint_vel(0)
quad_env.reset(qpos=init_qpos, qvel=init_qvel)

renderer = None
try:
    renderer = mujoco.Renderer(quad_env.mjModel, height=480, width=640)
except Exception:
    pass

qpos_list, qvel_list, images = [], [], []
for phase in range(ref.max_phase):
    sim_obs = quad_env._get_obs()
    target_pos = ref.get_joint_pos(phase)
    torque = (
        KP * (target_pos - sim_obs['qpos'][7:19])
        + KD * (ref.get_joint_vel(phase) - sim_obs['qvel'][6:18])
        + ref.get_feedforward_torque(phase)
    )
    torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
    for _ in range(substeps):
        sim_obs, _, _, _, _ = quad_env.step(action=torque)
    qpos_list.append(sim_obs['qpos'].copy())
    qvel_list.append(sim_obs['qvel'].copy())
    if renderer:
        try:
            renderer.update_scene(quad_env.mjData)
            images.append(renderer.render())
        except Exception:
            pass

if renderer:
    try: renderer.close()
    except: pass
quad_env.close()

qpos_traj = np.array(qpos_list)
qvel_traj_out = np.array(qvel_list)
n = len(qpos_traj)

sim_states = []
for i in range(min(n, state_traj.shape[0])):
    s, _ = sim_to_mpc(qpos_traj[i], qvel_traj_out[i])
    sim_states.append(s)
sim_traj = np.array(sim_states)
nn = min(state_traj.shape[0], sim_traj.shape[0])
pos_rms = np.sqrt(np.mean((state_traj[:nn, 0:3] - sim_traj[:nn, 0:3]) ** 2))
ori_rms = np.sqrt(np.mean((state_traj[:nn, 6:9] - sim_traj[:nn, 6:9]) ** 2))
joint_rms = np.sqrt(np.mean((state_traj[:nn, 12:24] - sim_traj[:nn, 12:24]) ** 2))

report = f'''
{'='*50}
BASELINE: PD + FEEDFORWARD (NO RL RESIDUALS)
{'='*50}
  Steps tracked: {n}/{state_traj.shape[0] - 1}
  Position RMS:    {pos_rms:.4f} m
  Orientation RMS: {ori_rms:.4f} rad
  Joint RMS:       {joint_rms:.4f} rad
{'='*50}
'''
print(report)
with open('$OUTPUT_DIR/baseline.log', 'w') as f:
    f.write(report)

if images:
    import imageio
    fps = int(1.0 / control_dt)
    imageio.mimsave('$OUTPUT_DIR/baseline.mp4', images, fps=fps)
    print(f'Video saved: $OUTPUT_DIR/baseline.mp4')
" 2>&1

echo "Done. Output: $OUTPUT_DIR/"
echo "Log:   $OUTPUT_DIR/baseline.log"
echo "Video: $OUTPUT_DIR/baseline.mp4"
