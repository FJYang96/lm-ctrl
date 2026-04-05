#!/bin/bash
# Evaluate the best model from the latest training run.
# Auto-finds the latest isaaclab_run_* in rl_isaac/trained_models/.
# Reads trajectory from rl_isaac/traj_config.sh (edit TRAJ_DIR and ITER_NUM there).
#
# Usage:
#   ./rl_isaac/evaluate_policy.sh
#
#   # Inside Docker:
#   docker run --gpus all -v $(pwd):/workspace/lm-ctrl --entrypoint bash \lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/evaluate_policy.sh

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
        "$IMAGE_NAME" /workspace/lm-ctrl/rl_isaac/evaluate_policy.sh "$@"
fi

cd /workspace/lm-ctrl 2>/dev/null || true

source "$SCRIPT_DIR/traj_config.sh"

rm -rf rl_isaac/policy_output/* 2>/dev/null
echo "Cleaned rl_isaac/policy_output/"

# ── Auto-find latest training run and its best model ──
LATEST_RUN=$(ls -td rl_isaac/trained_models/isaaclab_run_* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: no training runs found in rl_isaac/trained_models/"
    echo "  Run rl_isaac/run_smoke_test.sh first."
    exit 1
fi

MODEL_PATH="$LATEST_RUN/best_model"

# ── Validate model ──
if [ ! -f "$MODEL_PATH/checkpoint.pt" ]; then
    echo "ERROR: checkpoint.pt not found in $MODEL_PATH"
    exit 1
fi

# ── Validate trajectory files ──
for f in "$STATE_TRAJ" "$GRF_TRAJ" "$JOINT_VEL_TRAJ"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

OUTPUT_DIR="rl_isaac/policy_output/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# ── GPU selection: pick GPU with most free memory ──
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU $GPU_ID"

# ── Python setup ──
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    ISAAC_PYTHON="python3"
fi

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"

# ── Build contact sequence arg ──
CONTACT_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_FLAG="--contact-sequence $CONTACT_SEQ"
fi

# ── Print config ──
echo "============================================================"
echo "OPT-Mimic Policy Evaluation (Isaac Lab PhysX)"
echo "============================================================"
echo "  Model:      $MODEL_PATH"
echo "  Run:        $(basename "$LATEST_RUN")"
echo "  Trajectory: $TRAJ_DIR (iter $ITER_NUM)"
echo "  Contact:    $([ -f "$CONTACT_SEQ" ] && echo "yes" || echo "no (derived from GRF)")"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

# ── Run evaluation ──
$ISAAC_PYTHON -m rl_isaac.evaluate \
    --model-path "$MODEL_PATH" \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-video "$OUTPUT_DIR/eval_tracking.mp4" \
    --headless --enable_cameras \
    $CONTACT_FLAG 2>&1 | tee "$OUTPUT_DIR/eval.log"

echo ""
echo "============================================================"
echo "Evaluation complete"
echo "  Video: $OUTPUT_DIR/eval_tracking.mp4"
echo "  Log:   $OUTPUT_DIR/eval.log"
echo "============================================================"
