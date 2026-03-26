#!/bin/bash
# Isaac Lab OPT-Mimic: train, render best model video, evaluate, compare.
#
# Usage (from project root):
#   docker run --gpus all -v $(pwd):/workspace/lm-ctrl --entrypoint bash \
#       lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/run_smoke_test.sh [TIMESTEPS] [NUM_ENVS] [GPU_ID]

# end command: docker run:  docker ps -q | xargs -r docker stop; docker ps -qa | xargs -r docker rm
#start command: docker run --gpus all -v $(pwd):/workspace/lm-ctrl --entrypoint bash lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/run_smoke_test.sh
set -e

# Clean previous runs
rm -rf rl_isaac/trained_models/* 2>/dev/null
echo "Cleaned rl_isaac/trained_models/"

# GPU selection (default: GPU 1). Set via GPU_ID env var or defaults to 1.
GPU_ID=${GPU_ID:-2}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Lab Python not found. Run inside lm-ctrl-isaaclab Docker."
    echo ""
    echo "  docker run --gpus all -v \$(pwd):/workspace/lm-ctrl --entrypoint bash \\"
    echo "      lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/run_smoke_test.sh"
    exit 1
fi

TIMESTEPS=${1:-100000000}
NUM_ENVS=${2:-4096}

ITER_DIR="results/test"
STATE_TRAJ=${3:-$ITER_DIR/state_traj_iter_20.npy}
GRF_TRAJ=${4:-$ITER_DIR/grf_traj_iter_20.npy}
JOINT_VEL_TRAJ=${5:-$ITER_DIR/joint_vel_traj_iter_20.npy}
PLANNED_VIDEO=${6:-$ITER_DIR/planned_traj_iter_20.mp4}
CONTACT_SEQ=${7:-$ITER_DIR/contact_sequence_iter_20.npy}

RUN_TAG="isaaclab_run_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="rl_isaac/trained_models/$RUN_TAG"
LOG_FILE="$OUTPUT_DIR/experiment.log"

mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Trajectories: $STATE_TRAJ"
echo "Timesteps: $TIMESTEPS  Envs: $NUM_ENVS"

EXPERIMENT_START=$SECONDS

CONTACT_SEQ_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_SEQ_FLAG="--contact-sequence $CONTACT_SEQ"
fi

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"
export MUJOCO_GL=egl

# ── Step 1: Train (includes periodic videos every 1M steps + final best model video) ──
echo "[1/3] Training OPT-Mimic tracking policy (Isaac Lab PPO)..."
echo "      Videos rendered every 1M steps + final best model video."

$ISAAC_PYTHON -m rl_isaac.train \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-dir "$OUTPUT_DIR" \
    --total-timesteps "$TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --headless \
    $CONTACT_SEQ_FLAG

# ── Step 2: Evaluate best model (CPU MuJoCo rollout + tracking errors + video) ──
echo "[2/3] Evaluating best model (CPU MuJoCo)..."
MODEL_PATH="$OUTPUT_DIR/best_model"

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

args = ['evaluate',
    '--model-path', '$MODEL_PATH',
    '--state-traj', '$STATE_TRAJ',
    '--grf-traj', '$GRF_TRAJ',
    '--joint-vel-traj', '$JOINT_VEL_TRAJ',
    '--output-video', '$OUTPUT_DIR/rl_tracking.mp4',
]
if os.path.exists('$CONTACT_SEQ'):
    args += ['--contact-sequence', '$CONTACT_SEQ']
sys.argv = args
from rl_isaac.evaluate import main
main()
" 2>&1 | tee -a "$LOG_FILE"

# ── Step 3: Generate comparison frames ──
echo "[3/3] Generating comparison frames..."
$ISAAC_PYTHON -c "
import sys; sys.modules['glfw'] = type(sys)('glfw'); sys.modules['glfw.library'] = type(sys)('glfw.library')
sys.argv = ['generate_frames', '--planned', '$PLANNED_VIDEO', '--rl', '$OUTPUT_DIR/rl_tracking.mp4', '--output-dir', '$OUTPUT_DIR/comparison', '--num-frames', '20']
from rl_isaac.generate_frames import main; main()
" 2>/dev/null || echo "Frame generation skipped (no video or cv2 missing)"

EXPERIMENT_ELAPSED=$(( SECONDS - EXPERIMENT_START ))

echo "" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"
echo "Total experiment time: ${EXPERIMENT_ELAPSED}s" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "Isaac Lab smoke test complete in ${EXPERIMENT_ELAPSED}s"
echo "  Log:    $LOG_FILE"
echo "  Videos: $OUTPUT_DIR/runs/"
echo "  Best:   $OUTPUT_DIR/runs/best_model.mp4"
echo "============================================================"
