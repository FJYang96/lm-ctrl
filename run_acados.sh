#!/bin/bash
# Script to run the original Acados-based MPC

echo "🐳 Building Docker image..."
docker build -t quadruped-lm-ctrl .

echo "🚀 Running Acados-based MPC in Docker container..."
docker run --rm \
    -v $(pwd):/home/lm-ctrl \
    --workdir /home/lm-ctrl \
    quadruped-lm-ctrl \
    /opt/conda/bin/conda run -n quadruped_pympc_ros2_env bash -c "
        # Set library paths
        export LD_LIBRARY_PATH=/Quadruped-PyMPC/quadruped_pympc/acados/lib:/Quadruped-PyMPC/quadruped_pympc/acados/build/external/hpipm:\$LD_LIBRARY_PATH
        export ACADOS_SOURCE_DIR=/Quadruped-PyMPC/quadruped_pympc/acados
        export MUJOCO_GL=osmesa
        
        # Create results directory if it doesn't exist
        mkdir -p results
        
        # Run the original Acados implementation
        echo '📊 Running Acados-based trajectory optimization...'
        python main.py --solver acados
        
        echo '✅ Optimization complete! Check results/ directory for outputs.'
    "