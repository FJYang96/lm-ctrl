#!/bin/bash
# Smoke test: train RL tracking policy on results/ trajectory, then compare vs open-loop.
#
# Usage:
#   ./rl/run_smoke_test.sh [TIMESTEPS] [NUM_ENVS]
#
# Examples:
#   ./rl/run_smoke_test.sh              # 500k steps, 8 envs (quick test)
#   ./rl/run_smoke_test.sh 2000000 16   # 2M steps, 16 envs (full run)

set -e

TIMESTEPS=${1:-500000}
NUM_ENVS=${2:-8}
OUTPUT_DIR="rl/trained_models/smoke_test"

echo "=== OPT-Mimic Smoke Test ==="
echo "Timesteps: $TIMESTEPS"
echo "Parallel envs: $NUM_ENVS"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Train
echo "[1/2] Training tracking policy..."
python -m rl.train \
    --state-traj results/state_traj.npy \
    --grf-traj results/grf_traj.npy \
    --joint-vel-traj results/joint_vel_traj.npy \
    --output-dir "$OUTPUT_DIR" \
    --total-timesteps "$TIMESTEPS" \
    --num-envs "$NUM_ENVS"

# Step 2: Evaluate
echo ""
echo "[2/2] Comparing RL policy vs open-loop inverse dynamics..."

MODEL_PATH="$OUTPUT_DIR/best_model/best_model.zip"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="$OUTPUT_DIR/tracking_policy_final.zip"
fi

python -m rl.evaluate \
    --model-path "$MODEL_PATH" \
    --normalize-path "$OUTPUT_DIR/vec_normalize.pkl"
