#!/bin/bash
# This script runs inside the Docker container

# Activate conda environment
. /opt/conda/etc/profile.d/conda.sh
conda activate quadruped_pympc_ros2_env

# Set library paths
export LD_LIBRARY_PATH=/Quadruped-PyMPC/quadruped_pympc/acados/lib:/Quadruped-PyMPC/quadruped_pympc/acados/build/external/hpipm:$LD_LIBRARY_PATH

# Go to the mounted directory and run main.py
cd /home
python main.py