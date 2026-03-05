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
NUM_ENVS=${2:-256}
OUTPUT_DIR="rl/trained_models/smoke_test"

echo "=== OPT-Mimic Smoke Test (JAX/MJX) ==="
echo "Timesteps: $TIMESTEPS"
echo "Parallel envs: $NUM_ENVS"
echo "Output: $OUTPUT_DIR"
echo ""

# Clean all previous trained models, diagnostics, and old outputs
if [ -d "rl/trained_models" ]; then
    echo "Clearing rl/trained_models/..."
    rm -rf rl/trained_models
fi
rm -f rl/diagnostics.log
rm -rf results/comparison
rm -f results/rl_tracking.mp4

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
echo ""
echo "[2/3] Evaluating RL policy..."

MODEL_PATH="$OUTPUT_DIR/best_model"

MUJOCO_GL=egl python -m rl.evaluate \
    --model-path "$MODEL_PATH" \
    --output-video "$OUTPUT_DIR/rl_tracking.mp4"

# Step 3: Generate comparison frames
echo ""
echo "[3/3] Generating comparison frames..."
python -m rl.generate_frames \
    --rl "$OUTPUT_DIR/rl_tracking.mp4" \
    --num-frames 20 2>/dev/null || echo "Frame generation skipped (no video or cv2 missing)"

echo ""
echo "=== Smoke test complete ==="
