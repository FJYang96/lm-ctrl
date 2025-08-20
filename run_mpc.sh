#!/bin/bash

# Run the trajectory optimization in Docker
docker run -v /Users/aryan-roy/Desktop/lm-ctrl:/home --net=host \
  pympc bash -c \
  ". /opt/conda/etc/profile.d/conda.sh && \
   conda activate quadruped_pympc_ros2_env && \
   pip install lxml==4.9.3 && \
   export LD_LIBRARY_PATH=/Quadruped-PyMPC/quadruped_pympc/acados/lib:/Quadruped-PyMPC/quadruped_pympc/acados/build/external/hpipm && \
   export ACADOS_SOURCE_DIR=/Quadruped-PyMPC/quadruped_pympc/acados && \
   mkdir -p /home/results && \
   cd /home && \
   python main.py"