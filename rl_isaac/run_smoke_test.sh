#!/bin/bash
# Isaac Lab OPT-Mimic: train, render best model video, evaluate, compare.
#
# Reads trajectory from rl_isaac/traj_config.sh (edit TRAJ_DIR and ITER_NUM there).
#
# Usage (from project root):
#   docker run --gpus all -v $(pwd):/workspace/lm-ctrl --entrypoint bash \
#       lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/run_smoke_test.sh [TIMESTEPS] [NUM_ENVS]


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
        "$IMAGE_NAME" /workspace/lm-ctrl/rl_isaac/run_smoke_test.sh "$@"
fi

cd /workspace/lm-ctrl

source "$SCRIPT_DIR/traj_config.sh"

# Clean previous runs
#rm -rf rl_isaac/trained_models/* 2>/dev/null
echo "Cleaned rl_isaac/trained_models/"

# GPU selection: automatically pick the GPU with the most free memory.
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
GPU_ID=${GPU_ID:-0}
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU $GPU_ID (${GPU_FREE}MB free)"

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

RUN_TAG="isaaclab_run_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="rl_isaac/trained_models/$RUN_TAG"
LOG_FILE="$OUTPUT_DIR/experiment.log"

mkdir -p "$OUTPUT_DIR"

echo "Selected GPU $GPU_ID (${GPU_FREE}MB free)" >> "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Timesteps: $TIMESTEPS  Envs: $NUM_ENVS"

EXPERIMENT_START=$SECONDS

CONTACT_SEQ_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_SEQ_FLAG="--contact-sequence $CONTACT_SEQ"
fi

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"

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
    --headless --enable_cameras \
    $CONTACT_SEQ_FLAG

# ── Step 2: Evaluate best model (Isaac Lab PhysX) ──
echo "[2/3] Evaluating best model (Isaac Lab)..."
MODEL_PATH="$OUTPUT_DIR/best_model"

EVAL_CONTACT_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    EVAL_CONTACT_FLAG="--contact-sequence $CONTACT_SEQ"
fi

$ISAAC_PYTHON -m rl_isaac.evaluate \
    --model-path "$MODEL_PATH" \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-video "$OUTPUT_DIR/rl_tracking.mp4" \
    --headless --enable_cameras \
    $EVAL_CONTACT_FLAG 2>&1 | tee -a "$LOG_FILE"

# ── Step 3: Generate comparison frames ──
echo "[3/3] Generating comparison frames..."
if [ -f "$PLANNED_VIDEO" ]; then
    $ISAAC_PYTHON -m rl_isaac.generate_frames \
        --planned "$PLANNED_VIDEO" \
        --rl "$OUTPUT_DIR/rl_tracking.mp4" \
        --output-dir "$OUTPUT_DIR/comparison" \
        --num-frames 20 \
        2>/dev/null || echo "Frame generation failed (cv2 missing?)"
else
    echo "  Skipped: no planned video at $PLANNED_VIDEO"
fi

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
