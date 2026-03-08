#!/bin/bash
# Evaluate PD+feedforward baseline (no RL residuals) on a trajectory.
#
# Usage:
#   ./rl/run_baseline.sh [STATE_TRAJ] [GRF_TRAJ] [JOINT_VEL_TRAJ]
#
# Examples:
#   ./rl/run_baseline.sh                          # defaults: results/ trajectories
#   ./rl/run_baseline.sh path/to/state.npy path/to/grf.npy path/to/jvel.npy

set -e

# Ensure we run from the project root (needed for python -m rl.*)
cd "$(dirname "$0")/.."

ITER_DIR="results/llm_iterations/jump_180_degrees_entirely_and_land_1772942532"
STATE_TRAJ=${1:-$ITER_DIR/state_traj_iter_7.npy}
GRF_TRAJ=${2:-$ITER_DIR/grf_traj_iter_7.npy}
JOINT_VEL_TRAJ=${3:-$ITER_DIR/joint_vel_traj_iter_7.npy}

RUN_TAG="baseline_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="rl/trained_models/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Trajectories: $STATE_TRAJ"
echo ""

MUJOCO_GL=egl python -m rl.eval_baseline \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-video "$OUTPUT_DIR/baseline_tracking.mp4" \
    2>&1 | tee "$OUTPUT_DIR/baseline.log"

echo ""
echo "Done — video: $OUTPUT_DIR/baseline_tracking.mp4"
echo "       log:   $OUTPUT_DIR/baseline.log"
