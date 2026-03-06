#!/bin/bash
# Smoke test: train RL tracking policy on results/ trajectory, then compare vs open-loop.
#
# Usage:
#   ./rl/run_smoke_test.sh [TIMESTEPS] [NUM_ENVS]
#
# Examples:
#   ./rl/run_smoke_test.sh              # 200k steps, 256 envs (quick GPU test)
#   ./rl/run_smoke_test.sh 2000000 512  # 2M steps, 512 envs (full run)

set -e

TIMESTEPS=${1:-200000}
NUM_ENVS=${2:-1024}
OUTPUT_DIR="rl/trained_models/smoke_test"
LOG_FILE="$OUTPUT_DIR/experiment.log"

# Clean all previous trained models and old outputs
if [ -d "rl/trained_models" ]; then
    rm -rf rl/trained_models
fi
rm -f results/rl_tracking.mp4

# Ensure output directory exists (Python scripts also create it, but shell needs it for early logging)
mkdir -p "$OUTPUT_DIR"

EXPERIMENT_START=$SECONDS

# Step 1: Train
echo "[1/3] Training tracking policy (JAX PPO + MJX)..."
MUJOCO_GL=egl python -m rl.train \
    --state-traj results/state_traj.npy \
    --grf-traj results/grf_traj.npy \
    --joint-vel-traj results/joint_vel_traj.npy \
    --output-dir "$OUTPUT_DIR" \
    --total-timesteps "$TIMESTEPS" \
    --num-envs "$NUM_ENVS"

# Step 2: Evaluate
echo "[2/3] Evaluating RL policy..."
MODEL_PATH="$OUTPUT_DIR/best_model"
MUJOCO_GL=egl python -m rl.evaluate \
    --model-path "$MODEL_PATH" \
    --output-video "$OUTPUT_DIR/rl_tracking.mp4"

# Step 3: Generate comparison frames
echo "[3/3] Generating comparison frames..."
python -m rl.generate_frames \
    --planned results/planned_traj.mp4 \
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
