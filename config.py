from typing import Any

import numpy as np

import mpc.constraints as constr
from configs.experiments import BaseExperiment
from configs.robots import get_robot_data
from configs.robots.robot_data import RobotData
from mpc.config_complementarity import ComplementarityMPCConfig
from mpc.mpc_config import HoppingMPCConfig, MPCConfig

robot: str = "go2"
duration: float = 1.0
robot_data: RobotData = get_robot_data(robot)

# Select constraint mode: "standard" or "complementarity"
CONSTRAINT_MODE: str = "complementarity"  # Change this to switch modes

experiment: BaseExperiment = BaseExperiment(
    mu_ground=0.5,
    gravity_constant=9.81,
    duration=1.0,
    sim_dt=0.01,
    terrain="flat",
    initial_qpos=robot_data.initial_qpos,
    initial_qvel=robot_data.initial_qvel,
    render=True,
)

# Create MPC config based on selected mode
if CONSTRAINT_MODE == "complementarity":
    mpc_config: MPCConfig = ComplementarityMPCConfig(
        duration=duration,
        mpc_dt=0.02,
        pre_flight_stance_duration=0.3,
        flight_duration=0.4,
        q_base=np.diag([10, 10, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1.0,
        q_joint=np.eye(12) * 0.1,
        r_joint_vel=np.eye(12) * 1e-4,
        r_forces=np.eye(12) * 1e-5,
        q_terminal_base=np.eye(12) * 10.0,
        q_terminal_joint=np.eye(12) * 0.5,
        ref_state=None,
        ref_input=None,
        path_constraint_params={
            "COMPLEMENTARITY_EPS": 1e-3,  # Relaxation parameter for complementarity
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.04,
            "NO_SLIP_EPS": 0.01,
        },
        warmstart_state=None,
        warmstart_input=None,
        _contact_sequence=None,
    )
else:  # standard mode
    mpc_config = HoppingMPCConfig(
        duration=duration,
        mpc_dt=0.02,
        pre_flight_stance_duration=0.3,
        flight_duration=0.4,
        q_base=np.diag([10, 10, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1.0,
        q_joint=np.eye(12) * 0.1,
        r_joint_vel=np.eye(12) * 1e-4,
        r_forces=np.eye(12) * 1e-5,
        q_terminal_base=np.eye(12) * 10.0,
        q_terminal_joint=np.eye(12) * 0.5,
        ref_state=None,
        ref_input=None,
        path_constraints=[
            constr.friction_cone_constraints,
            constr.foot_height_constraints,
            constr.foot_velocity_constraints,
            constr.joint_limits_constraints,
            constr.input_limits_constraints,
        ],
        path_constraint_params={
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.04,
            "NO_SLIP_EPS": 0.01,
        },
        warmstart_state=None,
        warmstart_input=None,
        _contact_sequence=None,
    )

solver_config: dict[str, Any] = {
    "ipopt.print_level": 5,
    "print_time": True,
    "ipopt.max_iter": 1000,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 1e-3,
    "ipopt.mu_init": 1e-2,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.alpha_for_y": "primal",
    "ipopt.recalc_y": "yes",
}

# Simulation and logging parameters
plot_quantities = [
    "base_position",
    "base_linear_velocity",
    "base_orientation",
    "base_angular_velocity",
    "ground_reaction_forces",
]
