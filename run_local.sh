#!/bin/bash
# Run LLM pipeline locally (no Docker)
# Usage: ./run_local.sh "jump 180 degrees and land"
# Usage: ./run_local.sh "do a backflip" --max-iterations 10

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"<command>\" [options]"
    echo ""
    echo "Options:"
    echo "  --max-iterations N   Max LLM iterations (default: 20)"
    echo "  --results-dir PATH   Output directory (default: results/llm_iterations)"
    echo "  --no-slack           Use hard constraints"
    echo "  --verbose            Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 \"do a backflip\""
    echo "  $0 \"jump 180 degrees and land\" --max-iterations 5"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# PYTHON="/home/aryanroy/miniconda3/bin/python3"

# Non-interactive matplotlib
# conda init
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quadruped_pympc_ros2_jazzy_env
source .env
export MPLBACKEND=Agg
# Use EGL for headless OpenGL rendering (avoids gladLoadGL errors)
export MUJOCO_GL=egl
# Single-thread to avoid deadlocks in OpenBLAS/MUMPS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SCRIPT_DIR"
exec python llm_main.py "$@"
