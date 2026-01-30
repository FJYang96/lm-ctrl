#!/bin/bash
# Script to run LLM pipeline inside Docker container with full Acados environment

# Check if command argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"<command>\" [options]"
    echo "Examples:"
    echo "  $0 \"do a backflip\""
    echo "  $0 \"jump as high as possible\""
    echo "  $0 \"do a backflip\" --max-iterations 8"
    exit 1
fi

COMMAND="$1"
shift  # Remove first argument, keep the rest as options
OPTIONS="$@"

echo "🤖 LLM-Enhanced Quadruped Control in Docker"
echo "Command: '$COMMAND'"
echo "Options: $OPTIONS"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your Anthropic API key:"
    echo "ANTHROPIC_API_KEY=your_key_here"
    exit 1
fi

# Check if API key is set
if ! grep -q "ANTHROPIC_API_KEY=sk-" .env; then
    echo "❌ Error: ANTHROPIC_API_KEY not properly set in .env file"
    echo "Please edit .env and set your actual API key:"
    echo "ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here"
    exit 1
fi

# Create results directory on host BEFORE running Docker
mkdir -p results/llm_iterations

echo "🐳 Building Docker image..."
docker build -t quadruped-lm-ctrl .

echo "🚀 Running LLM pipeline in Docker container..."
docker run --rm \
    -v $(pwd):/home/lm-ctrl \
    --workdir /home/lm-ctrl \
    --env-file .env \
    quadruped-lm-ctrl \
    /opt/conda/bin/conda run -n quadruped_pympc_ros2_jazzy_env bash -c "
        # Set library paths for Acados
        export LD_LIBRARY_PATH=/Quadruped-PyMPC/quadruped_pympc/acados/lib:/Quadruped-PyMPC/quadruped_pympc/acados/build/external/hpipm:\$LD_LIBRARY_PATH
        export ACADOS_SOURCE_DIR=/Quadruped-PyMPC/quadruped_pympc/acados
        export MUJOCO_GL=osmesa

        # Create results directory
        mkdir -p results/llm_iterations

        # Install LLM dependencies
        echo '📦 Installing LLM dependencies...'
        pip install anthropic python-dotenv

        # Check environment
        echo '🔍 Checking environment...'
        python -c 'import casadi; print(f\"CasADi: {casadi.__version__}\")'
        python -c 'import anthropic; print(\"Anthropic: Available\")'
        python -c 'from mpc.dynamics.model import ACADOS_AVAILABLE; print(f\"Acados: {ACADOS_AVAILABLE}\")'

        # Run LLM pipeline
        echo '🧠 Starting LLM pipeline...'
        python llm_main.py \"$COMMAND\" $OPTIONS

        echo '✅ LLM pipeline complete! Check results/llm_iterations/ for outputs.'
    "

echo ""
echo "🎉 Done! Results are saved in results/llm_iterations/"
