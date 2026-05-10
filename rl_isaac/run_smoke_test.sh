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
rm -rf rl_isaac/trained_models/* 2>/dev/null
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

# ── Step 1: Train (includes periodic videos every 1M steps) ──
echo "[1/4] Training OPT-Mimic tracking policy (Isaac Lab PPO)..."
echo "      Videos rendered every 1M steps (DR-on snapshots, unseeded)."

$ISAAC_PYTHON -m rl_isaac.train \
    --state-traj "$STATE_TRAJ" \
    --grf-traj "$GRF_TRAJ" \
    --joint-vel-traj "$JOINT_VEL_TRAJ" \
    --output-dir "$OUTPUT_DIR" \
    --total-timesteps "$TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --headless --enable_cameras \
    $CONTACT_SEQ_FLAG

# ── Step 2: Evaluate best model (Isaac Lab PhysX) — clean deterministic ──
echo "[2/4] Evaluating best model (clean, deterministic)..."
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
    --output-json "$OUTPUT_DIR/eval_policy.json" \
    --headless --enable_cameras \
    $EVAL_CONTACT_FLAG 2>&1 | tee -a "$LOG_FILE"

# ── Step 3: Robustness sanity (20-seed DR sweep) ──
N_DR_SEEDS=${N_DR_SEEDS:-20}
echo ""
echo "[3/4] Robustness sweep ($N_DR_SEEDS seeds, DR enabled)..."
DR_OUT="$OUTPUT_DIR/dr_seeds"
mkdir -p "$DR_OUT"
for ((SEED=0; SEED<N_DR_SEEDS; SEED++)); do
    SEED_LOG="$DR_OUT/seed_${SEED}.stdout"
    $ISAAC_PYTHON -m rl_isaac.evaluate \
        --model-path "$MODEL_PATH" \
        --state-traj "$STATE_TRAJ" \
        --grf-traj "$GRF_TRAJ" \
        --joint-vel-traj "$JOINT_VEL_TRAJ" \
        --output-video "$DR_OUT/dr_seed_${SEED}.mp4" \
        --output-json  "$DR_OUT/dr_seed_${SEED}.json" \
        --enable-dr --seed "$SEED" \
        --headless --enable_cameras \
        $EVAL_CONTACT_FLAG > "$SEED_LOG" 2>&1
done
$ISAAC_PYTHON - <<PY 2>&1 | tee -a "$LOG_FILE"
import json, os
out_dir = "$DR_OUT"
clean_p = "$OUTPUT_DIR/eval_policy.json"
n_seeds = $N_DR_SEEDS
clean = json.load(open(clean_p)) if os.path.exists(clean_p) else None
n_pass = 0
n_total = 0
print()
print("=" * 70)
print("EVAL SUMMARY")
print("=" * 70)
if clean is not None:
    cs = "PASS" if (clean["frames_tracked"] == clean["max_phase"] and clean["termination_cause"] == "trunc") else "FAIL"
    print(f"clean (DR off, seed=0): {clean['frames_tracked']}/{clean['max_phase']}  cause={clean['termination_cause']}  rms_ori={clean['rms_ori']:.4f}  [{cs}]")
print()
print(f"DR sweep ({n_seeds} seeds):")
print(f"  {'seed':>4}  {'frames':>8}  {'cause':>10}  {'rms_ori':>8}  status")
fails = []
for s in range(n_seeds):
    p = f"{out_dir}/dr_seed_{s}.json"
    if not os.path.exists(p):
        print(f"  {s:>4}  MISSING")
        continue
    r = json.load(open(p))
    n_total += 1
    ok = (r["frames_tracked"] == r["max_phase"]) and r["termination_cause"] == "trunc"
    if ok: n_pass += 1
    else: fails.append(s)
    print(f"  {s:>4}  {r['frames_tracked']:>3}/{r['max_phase']:<3}  {str(r['termination_cause']):>10}  {r['rms_ori']:.4f}  {'PASS' if ok else 'FAIL'}")
print()
rate = (n_pass / n_total) if n_total else 0.0
# Wilson 95% CI lower bound for a binomial — quick heuristic robustness number.
import math
if n_total:
    z = 1.96
    p_hat = rate
    denom = 1 + z*z/n_total
    centre = (p_hat + z*z/(2*n_total)) / denom
    half = (z * math.sqrt(p_hat*(1-p_hat)/n_total + z*z/(4*n_total*n_total))) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    print(f"Robustness: {n_pass}/{n_total} pass  ({rate:.1%}, 95% CI [{lo:.1%}, {hi:.1%}])")
if fails:
    print(f"Failed seeds: {fails}")
print("=" * 70)
PY

# ── Step 4: Generate comparison frames ──
echo "[4/4] Generating comparison frames..."
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
echo "  Log:           $LOG_FILE"
echo "  Clean eval:    $OUTPUT_DIR/rl_tracking.mp4  (+ eval_policy.json)"
echo "  Training vids: $OUTPUT_DIR/runs/  (periodic, DR-on)"
echo "  DR sweep:      $OUTPUT_DIR/dr_seeds/  ($N_DR_SEEDS seeds)"
echo "============================================================"
