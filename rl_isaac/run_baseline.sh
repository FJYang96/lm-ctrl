#!/bin/bash
# PD + Feedforward baseline (no RL residuals).
# Shows tracking quality without any learned policy — just the PD controller + full ID.
# Uses Isaac Lab (PhysX) via evaluate.py --baseline.
#
# Reads trajectory from rl_isaac/traj_config.sh (edit TRAJ_DIR and ITER_NUM there).
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="lm-ctrl-isaaclab:latest"

# If not inside the container, ensure image exists and re-launch inside Docker
if [ ! -d "/workspace/isaaclab" ]; then
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Image $IMAGE_NAME not found — building..."
        docker build -t "$IMAGE_NAME" -f rl_isaac/Dockerfile.isaaclab .
    fi
    echo "Launching inside Docker..."
    exec docker run --gpus all -v "$(pwd)":/workspace/lm-ctrl --entrypoint bash \
        "$IMAGE_NAME" /workspace/lm-ctrl/rl_isaac/run_baseline.sh "$@"
fi

source "$SCRIPT_DIR/traj_config.sh"

# Clean previous baseline outputs
rm -rf rl_isaac/eval_output/* 2>/dev/null
echo "Cleaned rl_isaac/eval_output/"

# GPU selection: automatically pick the GPU with the most free memory.
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU $GPU_ID"

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Lab Python not found. Run inside lm-ctrl-isaaclab Docker."
    exit 1
fi

OUTPUT_DIR="rl_isaac/eval_output/baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"

CONTACT_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_FLAG="--contact-sequence $CONTACT_SEQ"
fi

echo "============================================================"
echo "PD+FF Baseline (zero RL residuals, Isaac Lab PhysX)"
echo "============================================================"
echo "  Trajectory: $TRAJ_DIR (iter $ITER_NUM)"
echo "  Contact:    $([ -f "$CONTACT_SEQ" ] && echo "yes" || echo "no (derived from GRF)")"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

$ISAAC_PYTHON -m rl_isaac.evaluate \
    --baseline \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-video "$OUTPUT_DIR/baseline.mp4" \
    --headless --enable_cameras \
    $CONTACT_FLAG 2>&1 | tee "$OUTPUT_DIR/baseline.log"

echo ""
echo "Done. Output: $OUTPUT_DIR/"
echo "  Log:   $OUTPUT_DIR/baseline.log"
echo "  Video: $OUTPUT_DIR/baseline.mp4"
