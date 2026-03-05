#!/bin/bash
# Fixed script to run the Opti-based MPC inside Docker container

echo "🐳 Building Docker image..."
docker build -t quadruped-lm-ctrl .

echo "🚀 Running Opti-based MPC in Docker container..."
docker run --rm \
    -v $(pwd):/home/lm-ctrl \
    --workdir /home/lm-ctrl \
    quadruped-lm-ctrl \
    /opt/conda/bin/conda run -n quadruped_pympc_ros2_jazzy_env bash -c "
        # Set library paths
        export LD_LIBRARY_PATH=/Quadruped-PyMPC/quadruped_pympc/acados/lib:/Quadruped-PyMPC/quadruped_pympc/acados/build/external/hpipm:\$LD_LIBRARY_PATH
        export ACADOS_SOURCE_DIR=/Quadruped-PyMPC/quadruped_pympc/acados
        export MUJOCO_GL=osmesa
        export MPLBACKEND=Agg

        # Prevent OpenBLAS/MUMPS thread deadlock in Docker
        export OMP_NUM_THREADS=4
        export OPENBLAS_NUM_THREADS=4
        export MKL_NUM_THREADS=4

        # Create results directory if it doesn't exist
        mkdir -p results

        # Check environment
        echo '🔍 Checking Python environment...'
        python -c 'import sys; print(f\"Python: {sys.executable}\")'
        python -c 'import casadi; print(f\"CasADi version: {casadi.__version__}\")'
        python -c 'import imageio; print(f\"ImageIO available\")'

        # Run the Opti-based implementation
        echo '📊 Running Opti-based trajectory optimization...'
        xvfb-run -a python main.py --solver opti

        echo '✅ Optimization complete! Check results/ directory for outputs.'
    "
