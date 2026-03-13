#!/bin/bash
# Smoke test: train RL tracking policy on a trajectory, then evaluate.
#
# Usage:
#   ./rl/run_smoke_test.sh [TIMESTEPS] [NUM_ENVS] [STATE_TRAJ] [GRF_TRAJ] [JOINT_VEL_TRAJ] [PLANNED_VIDEO]
#
# Examples:
#   ./rl/run_smoke_test.sh                          # defaults: 200k steps, results/ trajectories
#   ./rl/run_smoke_test.sh 500000 1024 path/to/state.npy path/to/grf.npy path/to/jvel.npy path/to/planned.mp4

set -e

# Reserve GPU 3 for other users
export CUDA_VISIBLE_DEVICES=0,1,2

TIMESTEPS=${1:-10000000}
NUM_ENVS=${2:-1024}
ITER_DIR="results/llm_iterations/jump_180_degrees_entirely_and_land_1772942532"
STATE_TRAJ=${3:-$ITER_DIR/state_traj_iter_7.npy}
GRF_TRAJ=${4:-$ITER_DIR/grf_traj_iter_7.npy}
JOINT_VEL_TRAJ=${5:-$ITER_DIR/joint_vel_traj_iter_7.npy}
PLANNED_VIDEO=${6:-$ITER_DIR/planned_traj_iter_7.mp4}
CONTACT_SEQ=${7:-$ITER_DIR/contact_sequence_iter_7.npy}

# Create a new timestamped run directory (never delete previous runs)
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="rl/trained_models/$RUN_TAG"
LOG_FILE="$OUTPUT_DIR/experiment.log"

mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Trajectories: $STATE_TRAJ"

EXPERIMENT_START=$SECONDS

# Step 1: Train
echo "[1/3] Training tracking policy (JAX PPO + MJX)..."
CONTACT_SEQ_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_SEQ_FLAG="--contact-sequence $CONTACT_SEQ"
fi

MUJOCO_GL=egl python -m rl.train \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-dir "$OUTPUT_DIR" \
    --total-timesteps "$TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    $CONTACT_SEQ_FLAG

# Step 2: Evaluate
echo "[2/3] Evaluating RL policy..."
MODEL_PATH="$OUTPUT_DIR/best_model"
MUJOCO_GL=egl python -m rl.evaluate \
    --model-path "$MODEL_PATH" \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-video "$OUTPUT_DIR/rl_tracking.mp4" \
    $CONTACT_SEQ_FLAG

# Step 3: Generate comparison frames
echo "[3/3] Generating comparison frames..."
python -m rl.generate_frames \
    --planned "$PLANNED_VIDEO" \
    --rl "$OUTPUT_DIR/rl_tracking.mp4" \
    --output-dir "$OUTPUT_DIR/comparison" \
    --num-frames 20 2>/dev/null || echo "Frame generation skipped (no video or cv2 missing)"

EXPERIMENT_ELAPSED=$(( SECONDS - EXPERIMENT_START ))

# Append total experiment time to the log
echo "" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"
echo "Total experiment time: ${EXPERIMENT_ELAPSED}s" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

echo ""
echo "Smoke test complete in ${EXPERIMENT_ELAPSED}s — log: $LOG_FILE"
