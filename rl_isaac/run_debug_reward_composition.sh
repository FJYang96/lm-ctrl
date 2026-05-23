#!/bin/bash
# Launch reward-composition visualization for a refine run.
#
# Usage:
#   ./rl_isaac/run_debug_reward_composition.sh <RUN_DIR> [ITER]
#
# Examples:
#   ./rl_isaac/run_debug_reward_composition.sh rl_isaac/refine_output/refine_iter_3_20260518_154456
#   ./rl_isaac/run_debug_reward_composition.sh rl_isaac/refine_output/refine_iter_3_20260518_154456 200

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="lm-ctrl-isaaclab:latest"

RUN_DIR="${1:-}"
ITER="${2:-}"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: run dir is required."
    echo "Usage: $0 <RUN_DIR> [ITER]"
    exit 1
fi

# If not inside the container, ensure image exists and re-launch inside Docker.
if [ ! -d "/workspace/isaaclab" ]; then
    DOCKER_RUN_DIR="$RUN_DIR"
    HOST_REPO="$(pwd)"
    case "$RUN_DIR" in
        "$HOST_REPO"/*)
            DOCKER_RUN_DIR="/workspace/lm-ctrl/${RUN_DIR#"$HOST_REPO"/}"
            ;;
    esac
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Image $IMAGE_NAME not found — building..."
        docker build -t "$IMAGE_NAME" -f rl_isaac/Dockerfile.isaaclab .
    fi
    echo "Launching inside Docker..."
    exec docker run --gpus all \
        -v "$(pwd)":/workspace/lm-ctrl \
        --entrypoint bash \
        "$IMAGE_NAME" /workspace/lm-ctrl/rl_isaac/run_debug_reward_composition.sh \
        "$DOCKER_RUN_DIR" "$ITER"
fi

cd /workspace/lm-ctrl

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Lab Python not found. Run inside lm-ctrl-isaaclab Docker."
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR not found: $RUN_DIR"
    exit 1
fi

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

APP_LAUNCH_FLAGS="--headless"

CMD=(
    "$ISAAC_PYTHON" -m rl_isaac.debug_reward_composition
    --run-dir "$RUN_DIR"
    "$APP_LAUNCH_FLAGS"
)

if [ -n "$ITER" ]; then
    CMD+=(--iter "$ITER")
fi

echo "============================================================"
echo "Reward composition debug"
echo "============================================================"
echo "  Run dir: $RUN_DIR"
if [ -n "$ITER" ]; then
    echo "  Iter:    $ITER"
else
    echo "  Iter:    auto-best from mppi_metrics.csv"
fi
echo "============================================================"

"${CMD[@]}"
