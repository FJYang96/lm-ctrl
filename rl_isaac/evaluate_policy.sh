#!/bin/bash
# Evaluate a trained OPT-Mimic policy against its reference trajectory.
#
# Usage:
#   ./rl_isaac/evaluate_policy.sh [model_path] [traj_dir] [output_dir]
#
# Defaults:
#   model_path  - rl_isaac/trained_models/isaaclab_run_20260322_232436/best_model
#   traj_dir    - results/jump  (iter 9, same as run_smoke_test.sh)
#   output_dir  - rl_isaac/eval_output/eval_<timestamp>
#
# Example:
#   ./rl_isaac/evaluate_policy.sh
#   ./rl_isaac/evaluate_policy.sh rl_isaac/trained_models/my_run/best_model results/jump
#
#   # Inside Docker:
#   docker run --gpus all -v $(pwd):/workspace/lm-ctrl --entrypoint bash \
#       lm-ctrl-isaaclab:latest /workspace/lm-ctrl/rl_isaac/evaluate_policy.sh

set -e

cd /workspace/lm-ctrl 2>/dev/null || true

rm -rf rl_isaac/policy_output/* 2>/dev/null
echo "Cleaned rl_isaac/policy_output/"

# ── Parse args with defaults ──
MODEL_PATH="${1:-rl_isaac/trained_models/isaaclab_run_20260327_025734/best_model}"
TRAJ_DIR="${2:-results/test}"
OUTPUT_DIR="${3:-rl_isaac/policy_output/eval_$(date +%Y%m%d_%H%M%S)}"

# ── Validate model ──
if [ ! -f "$MODEL_PATH/checkpoint.pt" ]; then
    echo "ERROR: checkpoint.pt not found in $MODEL_PATH"
    exit 1
fi

# ── Trajectory files (iter 9, same as run_smoke_test.sh) ──
ITER_NUM=20
STATE_TRAJ="$TRAJ_DIR/state_traj_iter_${ITER_NUM}.npy"
GRF_TRAJ="$TRAJ_DIR/grf_traj_iter_${ITER_NUM}.npy"
JOINT_VEL_TRAJ="$TRAJ_DIR/joint_vel_traj_iter_${ITER_NUM}.npy"
CONTACT_SEQ="$TRAJ_DIR/contact_sequence_iter_${ITER_NUM}.npy"

# ── Validate trajectory files ──
for f in "$STATE_TRAJ" "$GRF_TRAJ" "$JOINT_VEL_TRAJ"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

# ── Python setup ──
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [ ! -f "$ISAAC_PYTHON" ]; then
    ISAAC_PYTHON="python3"
fi

export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"
export MUJOCO_GL=egl

# ── Print config ──
echo "============================================================"
echo "OPT-Mimic Policy Evaluation"
echo "============================================================"
echo "  Model:      $MODEL_PATH"
echo "  Trajectory: iter $ITER_NUM from $TRAJ_DIR"
echo "  Contact:    $([ -f "$CONTACT_SEQ" ] && echo "yes" || echo "no (derived from GRF)")"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

# ── Build contact sequence arg ──
CONTACT_ARGS=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_ARGS="'--contact-sequence', '$CONTACT_SEQ',"
fi

# ── Run evaluation ──
$ISAAC_PYTHON -c "
import os, sys, types
os.environ['MUJOCO_GL'] = 'egl'
if 'mujoco.viewer' not in sys.modules:
    _fv = types.ModuleType('mujoco.viewer')
    _fv.Handle = type('Handle', (), {})
    sys.modules['mujoco.viewer'] = _fv
if 'glfw' not in sys.modules:
    _fg = types.ModuleType('glfw')
    _fg._glfw = True
    sys.modules['glfw'] = _fg
    sys.modules['glfw.library'] = types.ModuleType('glfw.library')

sys.argv = ['evaluate',
    '--model-path', '$MODEL_PATH',
    '--state-traj', '$STATE_TRAJ',
    '--grf-traj', '$GRF_TRAJ',
    '--joint-vel-traj', '$JOINT_VEL_TRAJ',
    '--output-video', '$OUTPUT_DIR/eval_tracking.mp4',
    $CONTACT_ARGS
]
from rl_isaac.evaluate import main
main()
"

echo ""
echo "============================================================"
echo "Evaluation complete"
echo "  Video: $OUTPUT_DIR/eval_tracking.mp4"
echo "============================================================"
