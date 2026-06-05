#!/bin/bash
# Plot upsampled reference vs Isaac rollout (1 kHz implicit PD by default).
#
# Usage:
#   ./rl_isaac/run_plot_feedforward_rollout.sh
#   ./rl_isaac/run_plot_feedforward_rollout.sh --no-feedforward
#   ./rl_isaac/run_plot_feedforward_rollout.sh --control-mode ff_invert --decimation 20

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="lm-ctrl-isaaclab:latest"

if [ ! -d "/workspace/isaaclab" ]; then
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Image $IMAGE_NAME not found — building..."
        docker build -t "$IMAGE_NAME" -f rl_isaac/Dockerfile.isaaclab .
    fi
    echo "Launching inside Docker..."
    exec docker run --gpus all \
        -v "$(pwd)":/workspace/lm-ctrl --entrypoint bash \
        "$IMAGE_NAME" /workspace/lm-ctrl/rl_isaac/run_plot_feedforward_rollout.sh "$@"
fi

cd /workspace/lm-ctrl
source "$SCRIPT_DIR/traj_config.sh"

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Lab Python not found. Run inside lm-ctrl-isaaclab Docker."
    exit 1
fi

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

RUN_TAG="ff_rollout_iter_${ITER_NUM}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="rl_isaac/physics_diagnostics_output"
LOG_FILE="$OUTPUT_DIR/${RUN_TAG}.log"
mkdir -p "$OUTPUT_DIR"

CONTACT_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_FLAG="--contact-sequence $CONTACT_SEQ"
fi

echo "============================================================"
echo "Feedforward rollout plots"
echo "============================================================"
echo "  Trajectory:  $TRAJ_DIR (iter $ITER_NUM)"
echo "  Output:      $OUTPUT_DIR/$RUN_TAG"
echo "  Log:         $LOG_FILE"
echo "============================================================"

$ISAAC_PYTHON -m rl_isaac.plot_feedforward_rollout \
    --traj-dir "$TRAJ_DIR" \
    --iter-num "$ITER_NUM" \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-dir "$OUTPUT_DIR" \
    --run-tag "$RUN_TAG" \
    --headless \
    $CONTACT_FLAG \
    "$@" 2>&1 | tee "$LOG_FILE"

cmd_status=${PIPESTATUS[0]}
RUN_DIR="$OUTPUT_DIR/$RUN_TAG"

if [ "$cmd_status" -ne 0 ]; then
    echo ""
    echo "Plot script failed with exit code $cmd_status"
    echo "See log: $LOG_FILE"
    exit "$cmd_status"
fi

echo ""
echo "Done. Plots in: $RUN_DIR"
