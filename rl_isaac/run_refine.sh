#!/bin/bash
# Refine nominal trajectory into an open-loop torque sequence via MPPI.
#
# Reads trajectory from rl_isaac/traj_config.sh (edit TRAJ_DIR and ITER_NUM there).
#
# Usage:
#   ./rl_isaac/run_refine.sh [MPPI_ITERS] [NUM_SAMPLES]

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="lm-ctrl-isaaclab:latest"

# If not inside the container, ensure image exists and re-launch inside Docker
if [ ! -d "/workspace/isaaclab" ]; then
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Image $IMAGE_NAME not found — building..."
        docker build -t "$IMAGE_NAME" -f rl_isaac/Dockerfile.isaaclab .
    fi
    echo "Launching inside Docker..."
    exec docker run --gpus all \
        -e RENDER_BEST_EVERY \
        -e REFINE_FORCE_HEADLESS \
        -v "$(pwd)":/workspace/lm-ctrl --entrypoint bash \
        "$IMAGE_NAME" /workspace/lm-ctrl/rl_isaac/run_refine.sh "$@"
fi

cd /workspace/lm-ctrl

source "$SCRIPT_DIR/traj_config.sh"

if [ ! -d "$TRAJ_DIR" ]; then
    echo "ERROR: TRAJ_DIR not found: $TRAJ_DIR"
    echo "Update rl_isaac/traj_config.sh to a valid iteration directory."
    exit 1
fi
for REQUIRED_TRAJ_FILE in "$STATE_TRAJ" "$GRF_TRAJ" "$JOINT_VEL_TRAJ"; do
    if [ ! -f "$REQUIRED_TRAJ_FILE" ]; then
        echo "ERROR: Required trajectory file missing: $REQUIRED_TRAJ_FILE"
        echo "Check TRAJ_DIR and ITER_NUM in rl_isaac/traj_config.sh"
        exit 1
    fi
done

# GPU selection: pick GPU with most free memory.
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
GPU_ID=${GPU_ID:-0}
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU $GPU_ID (${GPU_FREE}MB free)"

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Lab Python not found. Run inside lm-ctrl-isaaclab Docker."
    exit 1
fi

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

MPPI_ITERS=${1:-50}
NUM_SAMPLES=${2:-128}
RENDER_BEST_EVERY=${RENDER_BEST_EVERY:-0}
RUN_TAG="refine_iter_${ITER_NUM}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="rl_isaac/refine_output"
LOG_FILE="$OUTPUT_DIR/${RUN_TAG}.log"
mkdir -p "$OUTPUT_DIR"

# Render mode defaults:
# - no DISPLAY: run headless and rely on camera/offscreen frame capture.
# - with DISPLAY: keep GUI enabled unless REFINE_FORCE_HEADLESS=1.
APP_LAUNCH_FLAGS="--enable_cameras"
if [ -z "${DISPLAY:-}" ] || [ "${REFINE_FORCE_HEADLESS:-0}" = "1" ]; then
    APP_LAUNCH_FLAGS="--headless --enable_cameras"
fi

CONTACT_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_FLAG="--contact-sequence $CONTACT_SEQ"
fi

echo "============================================================"
echo "MPPI open-loop refinement"
echo "============================================================"
echo "  Trajectory:    $TRAJ_DIR (iter $ITER_NUM)"
echo "  MPPI iters:    $MPPI_ITERS"
echo "  Num samples:   $NUM_SAMPLES"
echo "  Render every:  $RENDER_BEST_EVERY (0=disabled, final video still saved)"
echo "  Output root:   $OUTPUT_DIR"
echo "  Log:           $LOG_FILE"
echo "  App flags:     $APP_LAUNCH_FLAGS"
echo "============================================================"

$ISAAC_PYTHON -m rl_isaac.refine \
    --traj-dir "$TRAJ_DIR" \
    --iter-num "$ITER_NUM" \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-dir "$OUTPUT_DIR" \
    --run-tag "$RUN_TAG" \
    --mppi-iters "$MPPI_ITERS" \
    --num-samples "$NUM_SAMPLES" \
    --render-best-every "$RENDER_BEST_EVERY" \
    $APP_LAUNCH_FLAGS \
    $CONTACT_FLAG 2>&1 | tee "$LOG_FILE"

cmd_status=${PIPESTATUS[0]}
if [ "$cmd_status" -ne 0 ]; then
    echo ""
    echo "Refine failed with exit code $cmd_status"
    echo "See log: $LOG_FILE"
    exit "$cmd_status"
fi

RUN_DIR="$OUTPUT_DIR/$RUN_TAG"
if [ ! -f "$RUN_DIR/run_summary.json" ]; then
    echo ""
    echo "Refine did not produce expected artifacts."
    echo "Missing: $RUN_DIR/run_summary.json"
    echo "See log: $LOG_FILE"
    exit 1
fi

echo ""
echo "Done. See $RUN_DIR/"
