FROM ubuntu:jammy

ARG TARGETARCH
RUN echo "Building for architecture: $TARGETARCH"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

    
# Install tkinter, pip, and gedit
RUN apt-get update && apt-get install -y python3-tk
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y gedit


# Install git
RUN apt-get update && apt-get install -y git
RUN apt install -y libosmesa6 libosmesa6-dev


# Install Miniforge
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git && \
    if [ "$TARGETARCH" = "amd64" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh; \
    else \
        echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    conda init bash && \
    conda clean -afy



RUN git clone https://github.com/iit-DLSLab/Quadruped-PyMPC.git
RUN cd Quadruped-PyMPC && git submodule update --init --recursive
RUN cd Quadruped-PyMPC && sed -i 's|git+https://github.com/iit-DLSLab/gym-quadruped.git|git+https://github.com/FJYang96/gym-quadruped.git|' pyproject.toml
# Create a conda environment with the specified name
RUN cd Quadruped-PyMPC/installation/mamba/integrated_gpu/ && conda env create -f mamba_environment_ros2_jazzy.yml

SHELL ["conda", "run", "-n", "quadruped_pympc_ros2_env", "/bin/bash", "-c"]

# Install acados
RUN cd Quadruped-PyMPC/quadruped_pympc/acados && mkdir -p build
RUN cd Quadruped-PyMPC/quadruped_pympc/acados/build && cmake -DACADOS_WITH_SYSTEM_BLASFEO:BOOL=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
RUN cd Quadruped-PyMPC/quadruped_pympc/acados/build && make install -j4
RUN pip install -e ./Quadruped-PyMPC/quadruped_pympc/acados/interfaces/acados_template
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/Quadruped-PyMPC/quadruped_pympc/acados/lib:/Quadruped-PyMPC/quadruped_pympc/acados/build/external/hpipm"' >> /root/.bashrc
RUN echo 'export ACADOS_SOURCE_DIR="/Quadruped-PyMPC/quadruped_pympc/acados"' >> /root/.bashrc
# Install Quadruped-PyMPC
RUN cd Quadruped-PyMPC && pip install -e .

# Pre-install the correct ARM64 Tera renderer to avoid runtime architecture issues
RUN if [ "$TARGETARCH" = "arm64" ]; then \
    mkdir -p /Quadruped-PyMPC/quadruped_pympc/acados/bin && \
    wget https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-arm64 -O /Quadruped-PyMPC/quadruped_pympc/acados/bin/t_renderer && \
    chmod +x /Quadruped-PyMPC/quadruped_pympc/acados/bin/t_renderer; \
fi

# Set the shell to bash and configure the shell prompt and aliases
RUN echo 'export PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;37m\]\u\[\033[00m\]@\[\033[01;32m\]\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\] "'  >> /root/.bashrc
RUN echo 'conda activate quadruped_pympc_ros2_env' >> /root/.bashrc

# Set the working directory and source the bashrc file
WORKDIR /home
RUN source /root/.bashrc
ENV MUJOCO_GL=osmesa
RUN pip install imageio[ffmpeg]