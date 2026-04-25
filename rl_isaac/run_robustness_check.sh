#!/bin/bash
# 5-seed DR-enabled robustness check on a trained policy.
#
# Reads MODEL_DIR, TRAJ_DIR, ITER_NUM the same way run_smoke_test.sh does
# (sourcing rl_isaac/traj_config.sh by default — pointing at the same
# trajectory the model was trained on). Runs evaluate.py 5 times with
# --enable-dr and --seed {0..4}, captures per-seed JSON + MP4, and prints
# a frames_tracked summary.
#
# Usage (from repo root):
#   bash rl_isaac/run_robustness_check.sh <MODEL_DIR> [N_SEEDS] [OUTPUT_DIR]
# Examples:
#   bash rl_isaac/run_robustness_check.sh \
#       rl_isaac/trained_models/isaaclab_run_20260425_175529
#   bash rl_isaac/run_robustness_check.sh \
#       rl_isaac/trained_models/isaaclab_run_20260425_175529 10 results/my_dr_check

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL_DIR> [N_SEEDS=5] [OUTPUT_DIR=<MODEL_DIR>/dr_seeds]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="lm-ctrl-isaaclab:latest"

# If not inside the container, ensure image exists and re-launch inside Docker
if [ ! -d "/workspace/isaaclab" ]; then
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Image $IMAGE_NAME not found — building..."
        docker build -t "$IMAGE_NAME" -f rl_isaac/Dockerfile.isaaclab .
    fi
    echo "Launching inside Docker..."
    exec docker run --rm --gpus all -v "$(pwd)":/workspace/lm-ctrl --entrypoint bash \
        "$IMAGE_NAME" /workspace/lm-ctrl/rl_isaac/run_robustness_check.sh "$@"
fi

cd /workspace/lm-ctrl

source "$SCRIPT_DIR/traj_config.sh"

MODEL_DIR="$1"
N_SEEDS="${2:-20}"
OUTPUT_DIR="${3:-${MODEL_DIR}/dr_seeds}"
mkdir -p "$OUTPUT_DIR"

if [ ! -d "$MODEL_DIR/best_model" ]; then
    echo "ERROR: $MODEL_DIR/best_model not found"
    exit 1
fi

GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
export PYTHONPATH="/workspace/lm-ctrl:${PYTHONPATH}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"

CONTACT_SEQ_FLAG=""
if [ -f "$CONTACT_SEQ" ]; then
    CONTACT_SEQ_FLAG="--contact-sequence $CONTACT_SEQ"
fi

echo "=========================================="
echo "Robustness check ($N_SEEDS seeds, DR enabled)"
echo "  model:    $MODEL_DIR/best_model"
echo "  traj:     $TRAJ_DIR  iter $ITER_NUM"
echo "  output:   $OUTPUT_DIR"
echo "  GPU:      $CUDA_VISIBLE_DEVICES"
echo "=========================================="

for ((S=0; S<N_SEEDS; S++)); do
    echo ""
    echo "--- seed $S ---"
    "$ISAAC_PYTHON" -m rl_isaac.evaluate \
        --model-path "$MODEL_DIR/best_model" \
        --state-traj "$STATE_TRAJ" \
        --grf-traj "$GRF_TRAJ" \
        --joint-vel-traj "$JOINT_VEL_TRAJ" \
        --output-video "$OUTPUT_DIR/dr_seed_${S}.mp4" \
        --output-json  "$OUTPUT_DIR/dr_seed_${S}.json" \
        --enable-dr --seed "$S" \
        --headless --enable_cameras \
        $CONTACT_SEQ_FLAG 2>&1 | grep -E "Steps tracked|Termination|RMS|terminated at step" | head -8
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
"$ISAAC_PYTHON" -c "
import json, os
n_pass = 0
n_total = 0
print(f'{\"seed\":>4}  {\"frames\":>8}  {\"cause\":>14}  {\"rms_pos\":>8}  {\"rms_ori\":>8}  {\"rms_joint\":>8}  status')
print('-'*78)
for s in range($N_SEEDS):
    p = f'$OUTPUT_DIR/dr_seed_{s}.json'
    if not os.path.exists(p):
        print(f'{s:>4}  MISSING')
        continue
    r = json.load(open(p))
    n_total += 1
    success = (r['frames_tracked'] == r['max_phase']) and r['termination_cause'] == 'trunc'
    if success: n_pass += 1
    mark = 'PASS' if success else 'FAIL'
    print(f'{s:>4}  {r[\"frames_tracked\"]:>3}/{r[\"max_phase\"]:<4}  {str(r[\"termination_cause\"]):>14}  {r[\"rms_pos\"]:.4f}  {r[\"rms_ori\"]:.4f}  {r[\"rms_joint\"]:.4f}  {mark}')
print()
print(f'{n_pass}/{n_total} successful  (>=4/5 = robustness OK per loop plan)')
"
